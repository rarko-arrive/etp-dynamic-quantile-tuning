-- Combined feature store (`$DQT_DATA_DIR/features.parquet`).
--
-- Population matches Tableau ETP monitoring (booked_etp_eligible ∩ booked_loads)
-- plus the analysis Committed Capacity exclusion, then inner-join latest ETP p05:
--   booked+ · not TONU · not Committed Capacity
--   · customer and carrier shipment charges > 250 · loadboard TotalCharges > 250
--   · miles > 0 · DRY/REEFER (V/VR/R) · single pickup and delivery
--   · no drop trailer · US-US · loadboard status not bounced/void/available/assigned/hold
--   · latest ETP p05 is not null.
--
-- Placeholders (str.format):
--   {start_date}  inclusive  booked_on_cst
--   {end_date}    exclusive  booked_on_cst
--
-- Canonical keys (dqt.score.constants.Cols — do not recompute in Python):
--   loadnumber  Cols.id
--   loaddate    Cols.time   Chicago timestamp string
--   date        Cols.date   Chicago calendar date (TO_DATE of booked_on_cst)
--
-- Run:  make features          # skip if parquet exists
--       make features FORCE=1  # rebuild
--       make smoke             # last 14 days → data/current

WITH
-- ---------------------------------------------------------------------------
-- Load attributes (SQL/queries/loads_base.sql)
-- ---------------------------------------------------------------------------
booked_loads AS (
    SELECT
        l.loadnumber,
        l.priced_to_make AS price_to_make,
        l.booked_on_cst,
        TO_DATE(l.booked_on_cst) AS booked_on_date,
        -- Canonical keys (Cols.id / Cols.time / Cols.date).
        -- booked_on_cst is America/Chicago wall time; date is that calendar day.
        TO_VARCHAR(l.booked_on_cst, 'YYYY-MM-DD HH24:MI:SS') AS loaddate,
        TO_DATE(l.booked_on_cst) AS date,
        l.created_on,
        l.created_on_cst,
        l.AVAILABLE_ON_FIRST_CST,
        l.PICKUP_APPT_LATEST_LOCAL,
        l.is_final_prebooked,
        vmm.booking_velocity,
        vmm.booked_to_cutoff,
        vmm.available_to_cutoff,
        l.Customer_charges_total,
        l.Carrier_charges_total,
        l.Customer_charges_total - l.Carrier_charges_total AS total_spread,
        l.Customer_Shipment_charges_total,
        l.Carrier_shipment_charges_total,
        l.customer_shipment_charges_total - l.carrier_shipment_charges_total AS shipment_spread,
        l.market_rate,
        l.internal_market_rate,
        l.market_comp,
        l.internal_cost_vs_mkt,
        l.origin_zone_group,
        l.dest_zone_group,
        l.origin_zone,
        l.dest_zone,
        l.origin_market_area_id,
        l.dest_market_area_id,
        l.origin_market_name,
        l.dest_market_name,
        l.origin_state,
        l.dest_state,
        l.origin_city,
        l.dest_city,
        l.origin_city_id,
        l.dest_city_id,
        l.origin_zip_five,
        l.dest_zip_five,
        l.market_lane,
        l.loaded_miles,
        l.booking_type,
        l.load_type,
        l.load_class,
        l.load_class_bucket,
        l.is_flamed,
        l.is_high_risk,
        l.is_high_value,
        l.customer_id,
        l.customer_name,
        l.carrier_id,
        l.carrier_name
    FROM core_data.core.loads AS l
    LEFT JOIN core_data.components.lod__status_mapping AS sm ON l.orderstatus = sm.order_status
    LEFT JOIN core_data.components.lod__velocity vmm ON l.loadnumber = vmm.loadnumber
    WHERE sm.is_covered = 1 -- Booked+
        AND l.is_tonu = False
        AND l.booking_type <> 'Committed Capacity'
        AND l.Customer_Shipment_charges_total > 250
        AND l.Carrier_shipment_charges_total > 250
        AND l.booked_on_cst >= '{start_date}'
        AND l.booked_on_cst < '{end_date}'
),
is_bounced AS (
    SELECT
        is_bad_bounce
        , bounce_date_time
        , loadnumber
    FROM (
        SELECT *, row_number() OVER (PARTITION BY LoadNumber ORDER BY bounce_date_time DESC) AS rown
        FROM core_data.core.bounces
    )
    WHERE rown = 1
),
load_base AS (
    SELECT
        bl.LoadNumber
        , bl.origin_zone_group
        , bl.dest_zone_group
        , bl.origin_zone
        , bl.dest_zone
        , bl.origin_market_name
        , bl.dest_market_name
        , bl.origin_state
        , bl.dest_state
        , bl.origin_city
        , bl.dest_city
        , bl.origin_city_id
        , bl.dest_city_id
        , bl.origin_zip_five
        , bl.dest_zip_five
        , bl.market_lane
        , bl.loaded_miles AS loadmiles
        , bl.booking_type
        , bl.load_type
        , bl.load_class
        , bl.load_class_bucket
        , lbb.weight
        , bl.is_flamed
        , bl.is_high_risk
        , bl.is_high_value
        , bl.customer_id
        , bl.customer_name
        , bl.carrier_id
        , bl.carrier_name
        , bl.created_on_cst
        , bl.available_on_first_cst
        , bl.booked_on_cst
        , bl.booked_on_date
        , bl.loaddate
        , bl.date
        , lbb.pickupapptearliest
        , bl.pickup_appt_latest_local
        , bl.available_to_cutoff
        , bl.booked_to_cutoff
        , bl.booking_velocity
        , bl.customer_charges_total
        , bl.carrier_charges_total
        , bl.total_spread
        , bl.customer_shipment_charges_total
        , bl.carrier_shipment_charges_total
        , bl.shipment_spread
        , bl.market_rate
        , bl.internal_market_rate
        , bl.price_to_make
        , bl.market_comp AS cost_vs_market
        , bl.internal_cost_vs_mkt AS internal_cost_vs_market
        , COALESCE(ib.is_bad_bounce, 0) AS is_bad_bounce
    FROM booked_loads AS bl
    LEFT JOIN is_bounced AS ib ON ib.loadnumber = bl.loadnumber
    INNER JOIN dapl_raw.accelerateprod.lod__loadboard_base lbb ON bl.loadnumber = lbb.loadnumber
    INNER JOIN dapl_raw.accelerateprod.ops__address AS oa ON oa.FacilityId = lbb.PickupEarlyFacilityId
    INNER JOIN dapl_raw.accelerateprod.ops__address AS da ON da.FacilityId = lbb.DeliveryLateFacilityId
    WHERE lbb.miles > 0
        AND lbb.TotalCharges > 250
        AND lbb.NumberOfPickups < 2
        AND lbb.NumberOfDeliveries < 2
        AND lbb.equipmenttype IN ('V', 'VR', 'R')
        AND lbb.LoadTypeDescription IN ('REEFER', 'DRY')
        AND (lbb.DropTrailer = 0 OR lbb.DropTrailer IS NULL)
        AND LOWER(lbb.orderstatus) NOT IN ('bounced', 'void', 'available', 'assigned', 'hold')
        AND oa.country IN ('US', 'USA', 'United States')
        AND da.country IN ('US', 'USA', 'United States')
),
-- ---------------------------------------------------------------------------
-- ETP / Weatherman / Lightning / displayed targets (SQL/queries/etp.sql)
-- Scoped to load_base so the heavy logs aren't scanned for ineligible ids.
-- ---------------------------------------------------------------------------
etp_letp AS (
    SELECT
        etp.*
        , row_number() OVER (PARTITION BY etp.LoadNumber ORDER BY etp.CreatedOn DESC) AS rown
    FROM dapl_raw.accelerateprod.lod__loadelitetruckpurchasing AS etp
    INNER JOIN load_base bl ON bl.loadnumber = etp.LoadNumber
    WHERE etp.createdon > '2022-09-18' -- Launch date
),
etp AS (
    SELECT
        lb.loadnumber
        , json_extract_path_text(etp_letp.percentiles, '"5"')::float AS P05
        , json_extract_path_text(etp_letp.percentiles, '"10"')::float AS P10
        , json_extract_path_text(etp_letp.percentiles, '"15"')::float AS P15
        , json_extract_path_text(etp_letp.percentiles, '"20"')::float AS P20
        , json_extract_path_text(etp_letp.percentiles, '"25"')::float AS P25
        , json_extract_path_text(etp_letp.percentiles, '"30"')::float AS P30
        , json_extract_path_text(etp_letp.percentiles, '"35"')::float AS P35
        , json_extract_path_text(etp_letp.percentiles, '"40"')::float AS P40
        , json_extract_path_text(etp_letp.percentiles, '"45"')::float AS P45
        , json_extract_path_text(etp_letp.percentiles, '"50"')::float AS P50
        , json_extract_path_text(etp_letp.percentiles, '"55"')::float AS P55
        , json_extract_path_text(etp_letp.percentiles, '"60"')::float AS P60
        , json_extract_path_text(etp_letp.percentiles, '"65"')::float AS P65
        , json_extract_path_text(etp_letp.percentiles, '"70"')::float AS P70
        , json_extract_path_text(etp_letp.percentiles, '"75"')::float AS P75
        , json_extract_path_text(etp_letp.percentiles, '"80"')::float AS P80
        , json_extract_path_text(etp_letp.percentiles, '"85"')::float AS P85
        , json_extract_path_text(etp_letp.percentiles, '"90"')::float AS P90
        , json_extract_path_text(etp_letp.percentiles, '"95"')::float AS P95
    FROM core_data.components.lod__base AS lb
    INNER JOIN etp_letp AS etp_letp ON etp_letp.loadnumber = lb.loadnumber AND etp_letp.rown = 1
),
eml AS (
    SELECT
        eml.*
        , row_number() OVER (PARTITION BY eml.LoadNumber ORDER BY eml.event_timestamp_cst DESC) AS rown
    FROM events_raw.etp.etp_model_logs AS eml
    INNER JOIN load_base bl ON bl.loadnumber = eml.LoadNumber
),
eml_last AS (
    SELECT
        eml.* EXCLUDE (rown)
    FROM eml
    WHERE eml.rown = 1
),
lightning AS (
    SELECT
        eml_last.loadnumber
        , eml_last.lightning_data:lightning_prediction::FLOAT AS lightning_prediction
    FROM eml_last
),
weatherman AS (
    SELECT
        eml_last.loadnumber
        , eml_last.quantiles:knn_5::FLOAT AS knn_5
        , eml_last.quantiles:knn_10::FLOAT AS knn_10
        , eml_last.quantiles:knn_15::FLOAT AS knn_15
        , eml_last.quantiles:knn_20::FLOAT AS knn_20
        , eml_last.quantiles:knn_25::FLOAT AS knn_25
        , eml_last.quantiles:knn_30::FLOAT AS knn_30
        , eml_last.quantiles:knn_35::FLOAT AS knn_35
        , eml_last.quantiles:knn_40::FLOAT AS knn_40
        , eml_last.quantiles:knn_45::FLOAT AS knn_45
        , eml_last.quantiles:knn_50::FLOAT AS knn_50
        , eml_last.quantiles:knn_55::FLOAT AS knn_55
        , eml_last.quantiles:knn_60::FLOAT AS knn_60
        , eml_last.quantiles:knn_65::FLOAT AS knn_65
        , eml_last.quantiles:knn_70::FLOAT AS knn_70
        , eml_last.quantiles:knn_75::FLOAT AS knn_75
        , eml_last.quantiles:knn_80::FLOAT AS knn_80
        , eml_last.quantiles:knn_85::FLOAT AS knn_85
        , eml_last.quantiles:knn_90::FLOAT AS knn_90
        , eml_last.quantiles:knn_95::FLOAT AS knn_95
    FROM eml_last
),
displayed_targets AS (
    SELECT
        loadnumber,
        CASE WHEN MAX(CASE WHEN displayid = 1 THEN etpvalue END) = 0 THEN NULL ELSE MAX(CASE WHEN displayid = 1 THEN etpvalue END) END AS ETP1,
        MAX(CASE WHEN displayid = 2 THEN etpvalue END) AS ETP2,
        MAX(CASE WHEN displayid = 3 THEN etpvalue END) AS ETP3,
        MAX(CASE WHEN displayid = 4 THEN etpvalue END) AS ETP4,
        CASE WHEN MAX(CASE WHEN displayid = 1 THEN percentile END) = 0 THEN NULL ELSE MAX(CASE WHEN displayid = 1 THEN percentile END) END AS T1_setting,
        MAX(CASE WHEN displayid = 2 THEN percentile END) AS T2_setting,
        MAX(CASE WHEN displayid = 3 THEN percentile END) AS T3_setting,
        MAX(CASE WHEN displayid = 4 THEN percentile END) AS T4_setting
    FROM (
        SELECT
            dt.loadnumber,
            dt.displayid,
            dt.etpvalue,
            dt.percentile::FLOAT AS percentile,
            ROW_NUMBER() OVER (PARTITION BY dt.loadnumber ORDER BY dt.modifiedon DESC) AS rn
        FROM DAPL_RAW.ACCELERATEPROD.LOD__DISPLAYEDTARGET AS dt
        INNER JOIN load_base bl ON bl.loadnumber = dt.loadnumber
    ) AS RankedRows
    WHERE rn <= 4
    GROUP BY loadnumber
),
-- ---------------------------------------------------------------------------
-- Origin / dest lat-lon (SQL/queries/load_coordinates.sql)
-- Inner on addresses here; the outer query left-joins so missing geocodes stay.
-- ---------------------------------------------------------------------------
coords AS (
    SELECT
        lbb.loadnumber
        , oa.geocode:coordinates[1]::float AS origin_lat
        , oa.geocode:coordinates[0]::float AS origin_lon
        , da.geocode:coordinates[1]::float AS dest_lat
        , da.geocode:coordinates[0]::float AS dest_lon
    FROM load_base AS lb
    INNER JOIN dapl_raw.accelerateprod.lod__loadboard_base AS lbb ON lbb.loadnumber = lb.loadnumber
    INNER JOIN dapl_raw.accelerateprod.ops__address AS oa ON oa.FacilityId = lbb.PickupEarlyFacilityId
    INNER JOIN dapl_raw.accelerateprod.ops__address AS da ON da.FacilityId = lbb.DeliveryLateFacilityId
),
-- ---------------------------------------------------------------------------
-- Shipped DQT dial (SQL/queries/dqt_alt_percentiles.sql)
-- One row per calendar day; left-joined on booked_on_date = valid_date.
-- ---------------------------------------------------------------------------
dqt_alt_raw AS (
    SELECT
        *
        , ROW_NUMBER() OVER (
            PARTITION BY valid_date
            ORDER BY snowflakeupdatedon DESC
        ) AS rown
    FROM DATA_SCIENCE.ETP_DYNAMIC_QUANTILE_TUNING.HISTORICAL_ETP_DYNAMIC_QUANTILES
),
dqt_alts AS (
    SELECT
        * EXCLUDE (rown, snowflakeupdatedon, alpha, time_window)
    FROM dqt_alt_raw
    WHERE rown = 1
        AND valid_date >= '{start_date}'
        AND valid_date < '{end_date}'
),
-- ---------------------------------------------------------------------------
-- DAT linehaul + SONAR orig/dest (SQL/queries/market.sql)
-- Left-joined on loadnumber so missing zone/DAT/SONAR stays null.
-- ---------------------------------------------------------------------------
DATDeltas AS (
    SELECT DISTINCT
        mda.ORIGCITY
        , mda.DESTCITY
        , mda.TRUCKTYPE
        , mda.DATE AS DATE
        , wow.DATE AS OldDate
        , mda.SPOTAVGLINEHAULRATE * mda.PCMILERPRACTICALMILEAGE AS dat_linehaul_total
        , wow.SPOTAVGLINEHAULRATE * wow.PCMILERPRACTICALMILEAGE AS dat_linehaul_total_lag1
        , mda.SPOTAVGLINEHAULRATE AS dat_linehaul_rate
        , wow.SPOTAVGLINEHAULRATE AS dat_linehaul_rate_lag1
        , mda.SPOTAVGLINEHAULRATE / NULLIF(wow.SPOTAVGLINEHAULRATE, 0) AS dat_linehaul_rate_percent_delta1
        , mda.SPOTAVGLINEHAULRATE * mda.PCMILERPRACTICALMILEAGE
            - wow.SPOTAVGLINEHAULRATE * wow.PCMILERPRACTICALMILEAGE AS dat_linehaul_total_delta1
        , mda.SPOTAVGLINEHAULRATE / NULLIF(two.SPOTAVGLINEHAULRATE, 0) AS dat_linehaul_rate_percent_delta2
        , mda.SPOTAVGLINEHAULRATE * mda.PCMILERPRACTICALMILEAGE
            - two.SPOTAVGLINEHAULRATE * two.PCMILERPRACTICALMILEAGE AS dat_linehaul_total_delta2
        , mda.SPOTAVGLINEHAULRATE / NULLIF(three.SPOTAVGLINEHAULRATE, 0) AS dat_linehaul_rate_percent_delta3
        , mda.SPOTAVGLINEHAULRATE * mda.PCMILERPRACTICALMILEAGE
            - three.SPOTAVGLINEHAULRATE * three.PCMILERPRACTICALMILEAGE AS dat_linehaul_total_delta3
    FROM DAPL_RAW.ACCELERATEPROD.PRL__MARKETDATA mda
    LEFT JOIN DAPL_RAW.ACCELERATEPROD.PRL__MARKETDATA wow
        ON wow.ORIGCITY = mda.ORIGCITY
        AND wow.DESTCITY = mda.DESTCITY
        AND wow.TRUCKTYPE = mda.TRUCKTYPE
        AND wow.DATE < DATEADD(DAY, -4, mda.DATE)
        AND wow.DATE >= DATEADD(DAY, -9, mda.DATE)
    LEFT JOIN DAPL_RAW.ACCELERATEPROD.PRL__MARKETDATA two
        ON two.ORIGCITY = mda.ORIGCITY
        AND two.DESTCITY = mda.DESTCITY
        AND two.TRUCKTYPE = mda.TRUCKTYPE
        AND two.DATE < DATEADD(DAY, -11, mda.DATE)
        AND two.DATE >= DATEADD(DAY, -16, mda.DATE)
    LEFT JOIN DAPL_RAW.ACCELERATEPROD.PRL__MARKETDATA three
        ON three.ORIGCITY = mda.ORIGCITY
        AND three.DESTCITY = mda.DESTCITY
        AND three.TRUCKTYPE = mda.TRUCKTYPE
        AND three.DATE < DATEADD(DAY, -18, mda.DATE)
        AND three.DATE >= DATEADD(DAY, -23, mda.DATE)
),
SONARDeltas AS (
    SELECT
        s.AIRPORTCODE
        , s.INDEXDATE
        , s.OTRI
        , s.ITRI
        , s.OTVI
        , s.ITVI
        , NULLIF(wow.OTRI, 0) AS otri_lag1
        , s.OTRI / NULLIF(wow.OTRI, 0) AS otri_percent_delta1
        , s.OTRI - wow.OTRI AS otri_delta1
        , s.OTRI / NULLIF(two.OTRI, 0) AS otri_percent_delta2
        , s.OTRI - two.OTRI AS otri_delta2
        , NULLIF(wow.ITVI, 0) AS itvi_lag1
        , s.ITVI / NULLIF(wow.ITVI, 0) AS itvi_percent_delta1
        , s.ITVI - wow.ITVI AS itvi_delta1
    FROM "DAPL_RAW"."THIRDPARTY"."SONAR" s
    INNER JOIN "DAPL_RAW"."THIRDPARTY"."SONAR" wow
        ON wow.AIRPORTCODE = s.AIRPORTCODE
        AND wow.INDEXDATE = DATEADD(DAY, -7, s.INDEXDATE)
    INNER JOIN "DAPL_RAW"."THIRDPARTY"."SONAR" two
        ON two.AIRPORTCODE = s.AIRPORTCODE
        AND two.INDEXDATE = DATEADD(DAY, -14, s.INDEXDATE)
),
DATDates AS (
    SELECT
        lb.loadnumber
        , MAX(d.DATE) AS DATDate
    FROM load_base AS lb
    INNER JOIN (SELECT DATE FROM DAPL_RAW.ACCELERATEPROD.PRL__MARKETDATA GROUP BY DATE) d
        ON d.DATE < lb.booked_on_date
    GROUP BY lb.loadnumber
),
market AS (
    SELECT
        feat.loadnumber
        , dd.dat_linehaul_total
        , dd.dat_linehaul_total_lag1
        , dd.dat_linehaul_rate
        , dd.dat_linehaul_rate_lag1
        , dd.dat_linehaul_rate_percent_delta1
        , dd.dat_linehaul_total_delta1
        , dd.dat_linehaul_rate_percent_delta2
        , dd.dat_linehaul_total_delta2
        , dd.dat_linehaul_rate_percent_delta3
        , dd.dat_linehaul_total_delta3
        , sdo.OTRI AS orig_otri
        , sdo.ITRI AS orig_itri
        , sdo.OTVI AS orig_otvi
        , sdo.ITVI AS orig_itvi
        , sdo.otri_percent_delta1 AS orig_otri_percent_delta1
        , sdo.otri_delta1 AS orig_otri_delta1
        , sdo.otri_percent_delta2 AS orig_otri_percent_delta2
        , sdo.otri_delta2 AS orig_otri_delta2
        , sdo.itvi_percent_delta1 AS orig_itvi_percent_delta1
        , sdo.itvi_delta1 AS orig_itvi_delta1
        , sdo.otri_lag1 AS orig_otri_lag1
        , sdo.itvi_lag1 AS orig_itvi_lag1
        , sdd.OTRI AS dest_otri
        , sdd.ITRI AS dest_itri
        , sdd.OTVI AS dest_otvi
        , sdd.ITVI AS dest_itvi
        , sdd.otri_percent_delta1 AS dest_otri_percent_delta1
        , sdd.otri_delta1 AS dest_otri_delta1
        , sdd.otri_percent_delta2 AS dest_otri_percent_delta2
        , sdd.otri_delta2 AS dest_otri_delta2
        , sdd.itvi_percent_delta1 AS dest_itvi_percent_delta1
        , sdd.itvi_delta1 AS dest_itvi_delta1
        , sdd.otri_lag1 AS dest_otri_lag1
        , sdd.itvi_lag1 AS dest_itvi_lag1
    FROM load_base AS feat
    INNER JOIN dapl_raw.accelerateprod.lod__loadboard_base lb ON lb.loadnumber = feat.loadnumber
    INNER JOIN dapl_raw.thirdparty.TranscoreMarketZone tmzo
        ON tmzo.THREEDIGITZIP = LEFT(lb.PickupEarlyPostalCode, 3)
    INNER JOIN dapl_raw.thirdparty.TranscoreMarketZone tmzd
        ON tmzd.THREEDIGITZIP = LEFT(lb.DeliveryLatePostalCode, 3)
    LEFT JOIN SONARDeltas sdo
        ON sdo.INDEXDATE = DATEADD(DAY, -1, feat.booked_on_date)
        AND LOWER(sdo.AIRPORTCODE) = LOWER(tmzo.AIRPORTCODE)
    LEFT JOIN SONARDeltas sdd
        ON sdd.INDEXDATE = DATEADD(DAY, -1, feat.booked_on_date)
        AND LOWER(sdd.AIRPORTCODE) = LOWER(tmzd.AIRPORTCODE)
    LEFT JOIN DATDates dts ON dts.loadnumber = feat.loadnumber
    LEFT JOIN DATDeltas dd
        ON LOWER(dd.OrigCity) = LOWER(tmzo.CENTROIDCITY)
        AND LOWER(dd.DestCity) = LOWER(tmzd.CENTROIDCITY)
        AND dd.TruckType = CASE
            WHEN lb.LoadTypeId = 1 THEN 'V'
            WHEN lb.loadtypeid IN (2, 9) THEN 'R'
            ELSE NULL
        END
        AND dd.DATE = dts.DATDate
    WHERE lb.loadtypeid IN (1, 2, 9)
)
SELECT
    lb.*
    , l.lightning_prediction
    , w.* EXCLUDE (loadnumber)
    , etp.* EXCLUDE (loadnumber)
    , CASE WHEN lb.Carrier_shipment_charges_total < etp.P05 THEN 1 ELSE 0 END AS ETP05_attainment
    , CASE WHEN lb.Carrier_shipment_charges_total < etp.P10 THEN 1 ELSE 0 END AS ETP10_attainment
    , CASE WHEN lb.Carrier_shipment_charges_total < etp.P15 THEN 1 ELSE 0 END AS ETP15_attainment
    , CASE WHEN lb.Carrier_shipment_charges_total < etp.P20 THEN 1 ELSE 0 END AS ETP20_attainment
    , CASE WHEN lb.Carrier_shipment_charges_total < etp.P25 THEN 1 ELSE 0 END AS ETP25_attainment
    , CASE WHEN lb.Carrier_shipment_charges_total < etp.P30 THEN 1 ELSE 0 END AS ETP30_attainment
    , CASE WHEN lb.Carrier_shipment_charges_total < etp.P35 THEN 1 ELSE 0 END AS ETP35_attainment
    , CASE WHEN lb.Carrier_shipment_charges_total < etp.P40 THEN 1 ELSE 0 END AS ETP40_attainment
    , CASE WHEN lb.Carrier_shipment_charges_total < etp.P45 THEN 1 ELSE 0 END AS ETP45_attainment
    , CASE WHEN lb.Carrier_shipment_charges_total < etp.P50 THEN 1 ELSE 0 END AS ETP50_attainment
    , CASE WHEN lb.Carrier_shipment_charges_total < etp.P55 THEN 1 ELSE 0 END AS ETP55_attainment
    , CASE WHEN lb.Carrier_shipment_charges_total < etp.P60 THEN 1 ELSE 0 END AS ETP60_attainment
    , CASE WHEN lb.Carrier_shipment_charges_total < etp.P65 THEN 1 ELSE 0 END AS ETP65_attainment
    , CASE WHEN lb.Carrier_shipment_charges_total < etp.P70 THEN 1 ELSE 0 END AS ETP70_attainment
    , CASE WHEN lb.Carrier_shipment_charges_total < etp.P75 THEN 1 ELSE 0 END AS ETP75_attainment
    , CASE WHEN lb.Carrier_shipment_charges_total < etp.P80 THEN 1 ELSE 0 END AS ETP80_attainment
    , CASE WHEN lb.Carrier_shipment_charges_total < etp.P85 THEN 1 ELSE 0 END AS ETP85_attainment
    , CASE WHEN lb.Carrier_shipment_charges_total < etp.P90 THEN 1 ELSE 0 END AS ETP90_attainment
    , CASE WHEN lb.Carrier_shipment_charges_total < etp.P95 THEN 1 ELSE 0 END AS ETP95_attainment
    , dt.T1_setting AS T1_setting
    , dt.T2_setting AS T2_setting
    , dt.T3_setting AS T3_setting
    , dt.T4_setting AS T4_setting
    , dt.ETP1 AS T1
    , dt.ETP2 AS T2
    , dt.ETP3 AS T3
    , dt.ETP4 AS T4
    , CASE WHEN lb.Carrier_shipment_charges_total < dt.ETP1 THEN 1 ELSE 0 END AS T1_attainment
    , CASE WHEN lb.Carrier_shipment_charges_total < dt.ETP2 THEN 1 ELSE 0 END AS T2_attainment
    , CASE WHEN lb.Carrier_shipment_charges_total < dt.ETP3 THEN 1 ELSE 0 END AS T3_attainment
    , CASE WHEN lb.Carrier_shipment_charges_total < dt.ETP4 THEN 1 ELSE 0 END AS T4_attainment
    , c.origin_lat
    , c.origin_lon
    , c.dest_lat
    , c.dest_lon
    , dqt.* EXCLUDE (valid_date)
    , mkt.* EXCLUDE (loadnumber)
FROM load_base AS lb
INNER JOIN etp ON etp.loadnumber = lb.loadnumber AND etp.p05 IS NOT NULL
LEFT JOIN lightning AS l ON lb.loadnumber = l.loadnumber
LEFT JOIN weatherman AS w ON lb.loadnumber = w.loadnumber
LEFT JOIN displayed_targets AS dt ON lb.loadnumber = dt.loadnumber
LEFT JOIN coords AS c ON lb.loadnumber = c.loadnumber
LEFT JOIN dqt_alts AS dqt ON TO_DATE(dqt.valid_date) = lb.booked_on_date
LEFT JOIN market AS mkt ON mkt.loadnumber = lb.loadnumber
