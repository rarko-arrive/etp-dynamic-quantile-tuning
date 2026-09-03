/*
Purpose: ETP shipment funnel — load-grain stage flags (Stages 0–6).
Owner: DQT
Params: {ship_date_start}, {ship_date_end} — inclusive ship_date_coalesce window
Consumers: scripts/export_funnel_stages.py → mart/funnel/funnel_stages.parquet
Link: .ai/plans/etp-slider/funnel-improvement-plan.md
*/

WITH base AS (
    SELECT
        l.loadnumber,
        TO_VARCHAR(DATE_TRUNC('month', l.ship_date_coalesce), 'YYYY-MM') AS ship_month
    FROM core_data.core.loads AS l
    WHERE l.order_status_group = 'Covered'
      AND l.ship_date_coalesce BETWEEN '{ship_date_start}' AND '{ship_date_end}'
),

booked_truck_load AS (
    SELECT l.loadnumber, 1 AS stage_booked_tl
    FROM core_data.core.loads AS l
    LEFT JOIN dapl_raw.accelerateprod.lod__loadboard_base AS lb
        ON l.loadnumber = lb.loadnumber
    LEFT JOIN core_data.components.lod__status_mapping AS sm
        ON l.orderstatus = sm.order_status
    WHERE sm.is_covered = 1
      AND lb.totalcharges > 250
      AND LOWER(lb.orderstatus) NOT IN ('bounced', 'void', 'available', 'assigned', 'hold')
      AND LOWER(lb.loadtypedescription) NOT IN (
          'imdl', 'ltl', 'drayage', 'power only', 'partial', 'air'
      )
),

booked_etp_eligible AS (
    SELECT l.loadnumber, 1 AS stage_etp_product_eligible
    FROM core_data.core.loads AS l
    LEFT JOIN dapl_raw.accelerateprod.lod__loadboard_base AS lb
        ON l.loadnumber = lb.loadnumber
    LEFT JOIN dapl_raw.accelerateprod.ops__address AS o
        ON lb.pickupearlyfacilityid = o.facilityid
    LEFT JOIN dapl_raw.accelerateprod.ops__address AS d
        ON lb.deliverylatefacilityid = d.facilityid
    LEFT JOIN core_data.components.lod__status_mapping AS sm
        ON l.orderstatus = sm.order_status
    WHERE sm.is_covered = 1
      AND LOWER(lb.orderstatus) NOT IN ('bounced', 'void', 'available', 'assigned', 'hold')
      AND lb.miles > 0
      AND lb.totalcharges > 250
      AND lb.numberofpickups < 2
      AND lb.numberofdeliveries < 2
      AND lb.equipmenttype IN ('V', 'VR', 'R')
      AND lb.loadtypedescription IN ('REEFER', 'DRY')
      AND (lb.droptrailer = 0 OR lb.droptrailer IS NULL)
      AND o.country IN ('US', 'USA', 'United States')
      AND d.country IN ('US', 'USA', 'United States')
      AND (l.is_tonu = 0 OR l.is_tonu IS NULL)
),

etp_percentiles AS (
    SELECT loadnumber, is_etp_eligible AS stage_etp_finance_eligible
    FROM core_data.components.lod__etp_percentiles
),

prebook_stats AS (
    SELECT loadnumber, is_prebook_eligible AS stage_prebook_eligible
    FROM (
        SELECT
            *,
            RANK() OVER (
                PARTITION BY loadnumber ORDER BY snapshot_createdon_cst DESC
            ) AS row_num
        FROM core_data.components.lod__prebook
    ) AS a
    WHERE row_num = 1
),

load_base AS (
    SELECT
        l.loadnumber,
        CONVERT_TIMEZONE(
            'America/Chicago', 'UTC',
            l.available_on_first_cst::TIMESTAMP_NTZ
        ) AS made_available_utc,
        CONVERT_TIMEZONE(
            l.origin_iana_timezone_name, 'UTC',
            l.pickup_appt_latest_local::TIMESTAMP_NTZ
        ) AS pickup_appt_latest_utc,
        l.pickup_appt_latest_local::TIMESTAMP_NTZ AS pickup_appt_latest_local
    FROM core_data.core.loads AS l
    WHERE l.available_on_first_cst IS NOT NULL
      AND l.pickup_appt_latest_local IS NOT NULL
      AND l.origin_iana_timezone_name IS NOT NULL
      AND l.ship_date_coalesce BETWEEN '{ship_date_start}' AND '{ship_date_end}'
      AND l.order_status_group = 'Covered'
),

load_filtered AS (
    SELECT
        loadnumber,
        made_available_utc,
        pickup_appt_latest_utc,
        pickup_appt_latest_local,
        DATEADD('hour', -24, pickup_appt_latest_utc) AS ts_24hr_remaining_utc
    FROM load_base
    WHERE DATEDIFF('minute', made_available_utc, pickup_appt_latest_utc) > 72 * 60
),

model AS (
    SELECT
        m.loadnumber,
        m.event_timestamp_utc::TIMESTAMP_NTZ AS snapshot_utc,
        PARSE_JSON(m.lightning_data):"features_used":"PickupApptLatest"::TIMESTAMP_NTZ
            AS blob_pickup_appt_local
    FROM events_raw.etp.etp_model_logs AS m
    INNER JOIN load_base AS l ON l.loadnumber = m.loadnumber
    WHERE m.etp_output IS NOT NULL
      AND PARSE_JSON(m.etp_output):"Percentiles" IS NOT NULL
),

appt_validation AS (
    SELECT
        f.loadnumber,
        COUNT_IF(
            mo.blob_pickup_appt_local::DATE <> f.pickup_appt_latest_local::DATE
        ) AS mismatch_count
    FROM load_filtered AS f
    JOIN model AS mo
        ON mo.loadnumber = f.loadnumber
       AND mo.snapshot_utc >= f.made_available_utc
       AND mo.snapshot_utc <= f.ts_24hr_remaining_utc
    GROUP BY f.loadnumber
),

load_final AS (
    SELECT f.*
    FROM load_filtered AS f
    LEFT JOIN appt_validation AS a ON a.loadnumber = f.loadnumber
    WHERE COALESCE(a.mismatch_count, 0) = 0
),

priced AS (
    SELECT DISTINCT f.loadnumber
    FROM load_final AS f
    JOIN model AS mo
        ON mo.loadnumber = f.loadnumber
       AND mo.snapshot_utc >= f.made_available_utc
       AND mo.snapshot_utc <= f.ts_24hr_remaining_utc
),

slider_flags AS (
    SELECT loadnumber, 1 AS stage_slider_cohort FROM load_filtered
),

stable_flags AS (
    SELECT loadnumber, 1 AS stage_stable_appt FROM load_final
),

priced_flags AS (
    SELECT loadnumber, 1 AS stage_etp_priced FROM priced
),

joined AS (
    SELECT
        b.loadnumber,
        b.ship_month,
        1 AS stage_all_covered,
        COALESCE(btl.stage_booked_tl, 0) AS stage_booked_tl,
        COALESCE(ee.stage_etp_product_eligible, 0) AS stage_etp_product_eligible,
        COALESCE(ep.stage_etp_finance_eligible, 0) AS stage_etp_finance_eligible,
        COALESCE(pb.stage_prebook_eligible, 0) AS stage_prebook_eligible,
        COALESCE(sf.stage_slider_cohort, 0) AS stage_slider_cohort,
        COALESCE(st.stage_stable_appt, 0) AS stage_stable_appt,
        COALESCE(pr.stage_etp_priced, 0) AS stage_etp_priced
    FROM base AS b
    LEFT JOIN booked_truck_load AS btl ON b.loadnumber = btl.loadnumber
    LEFT JOIN booked_etp_eligible AS ee ON b.loadnumber = ee.loadnumber
    LEFT JOIN etp_percentiles AS ep ON b.loadnumber = ep.loadnumber
    LEFT JOIN prebook_stats AS pb ON b.loadnumber = pb.loadnumber
    LEFT JOIN slider_flags AS sf ON b.loadnumber = sf.loadnumber
    LEFT JOIN stable_flags AS st ON b.loadnumber = st.loadnumber
    LEFT JOIN priced_flags AS pr ON b.loadnumber = pr.loadnumber
)

SELECT
    loadnumber,
    ship_month,
    stage_all_covered::BOOLEAN AS stage_all_covered,
    stage_booked_tl::BOOLEAN AS stage_booked_tl,
    stage_etp_product_eligible::BOOLEAN AS stage_etp_product_eligible,
    stage_etp_finance_eligible::BOOLEAN AS stage_etp_finance_eligible,
    stage_prebook_eligible::BOOLEAN AS stage_prebook_eligible,
    stage_slider_cohort::BOOLEAN AS stage_slider_cohort,
    stage_stable_appt::BOOLEAN AS stage_stable_appt,
    stage_etp_priced::BOOLEAN AS stage_etp_priced
FROM joined
ORDER BY loadnumber;
