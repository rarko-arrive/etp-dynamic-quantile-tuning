/*
Purpose: ETP shipment funnel — monthly eligibility waterfall (Stages 0–3).
Owner: DQT
Params: ship_date_start, ship_date_end (via query_sf / notebook)
Consumers: notebooks/etp-slider/etp-funnel.ipynb; mart/funnel/funnel_stages.parquet export
Link: .ai/plans/etp-slider/funnel-improvement-plan.md
*/

WITH base AS (
    SELECT
        l.loadnumber,
        DATE_TRUNC('month', l.ship_date_coalesce)::DATE AS ship_month
    FROM core_data.core.loads AS l
    WHERE l.order_status_group = 'Covered'
      AND l.ship_date_coalesce BETWEEN '{ship_date_start}' AND '{ship_date_end}'
),
booked_truck_load AS (
    SELECT
        l.loadnumber,
        1 AS stage_booked_tl
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
    SELECT
        l.loadnumber,
        1 AS stage_etp_product_eligible
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
    SELECT
        loadnumber,
        is_etp_eligible AS stage_etp_finance_eligible
    FROM core_data.components.lod__etp_percentiles
),
prebook_stats AS (
    SELECT
        loadnumber,
        is_prebook_eligible AS stage_prebook_eligible
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
joined AS (
    SELECT
        b.loadnumber,
        b.ship_month,
        1 AS stage_all_covered,
        COALESCE(btl.stage_booked_tl, 0) AS stage_booked_tl,
        COALESCE(ee.stage_etp_product_eligible, 0) AS stage_etp_product_eligible,
        COALESCE(ep.stage_etp_finance_eligible, 0) AS stage_etp_finance_eligible,
        COALESCE(pb.stage_prebook_eligible, 0) AS stage_prebook_eligible
    FROM base AS b
    LEFT JOIN booked_truck_load AS btl ON b.loadnumber = btl.loadnumber
    LEFT JOIN booked_etp_eligible AS ee ON b.loadnumber = ee.loadnumber
    LEFT JOIN etp_percentiles AS ep ON b.loadnumber = ep.loadnumber
    LEFT JOIN prebook_stats AS pb ON b.loadnumber = pb.loadnumber
)
SELECT
    ship_month,
    'all_covered' AS stage,
    COUNT(DISTINCT loadnumber) AS n_loads
FROM joined
WHERE stage_all_covered = 1
GROUP BY 1, 2

UNION ALL

SELECT ship_month, 'booked_tl', COUNT(DISTINCT loadnumber)
FROM joined WHERE stage_booked_tl = 1 GROUP BY 1, 2

UNION ALL

SELECT ship_month, 'etp_product_eligible', COUNT(DISTINCT loadnumber)
FROM joined WHERE stage_etp_product_eligible = 1 GROUP BY 1, 2

UNION ALL

SELECT ship_month, 'etp_finance_eligible', COUNT(DISTINCT loadnumber)
FROM joined WHERE stage_etp_finance_eligible = 1 GROUP BY 1, 2

UNION ALL

SELECT ship_month, 'prebook_eligible', COUNT(DISTINCT loadnumber)
FROM joined WHERE stage_prebook_eligible = 1 GROUP BY 1, 2

ORDER BY ship_month, stage;
