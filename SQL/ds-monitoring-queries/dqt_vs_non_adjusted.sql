/*
Purpose: ETP attainment vs DQT-adjusted attainment comparison.
Owner: Steve
Params: none
Consumers: Tableau; notebooks/ds-monitoring.ipynb
Link: https://prod-useast-b.online.tableau.com/#/site/arrivelogistics/views/ETPAttainmentvsDQTAttainment/Dashboard1
*/

WITH percentiles as (
select * exclude (rown)
from (
select etp.event_timestamp_utc, lc.booked_on_cst, etp.loadnumber, lc.Carrier_charges_total as TotalCosts
, json_extract_path_text(ETP_OUTPUT, 'Percentiles."5"')::float as final_p5
, json_extract_path_text(QUANTILES, 'knn_5')::float as orig_p5
, json_extract_path_text(ALT_QUANTILES, 'knn_5.value')::float as alt_p5
, json_extract_path_text(ALT_QUANTILES, 'knn_5.quantile')::float as alt_p5_quantile
, json_extract_path_text(ETP_OUTPUT, 'Percentiles."10"')::float as final_p10
, json_extract_path_text(QUANTILES, 'knn_10')::float as orig_p10
, json_extract_path_text(ALT_QUANTILES, 'knn_10.value')::float as alt_p10
, json_extract_path_text(ALT_QUANTILES, 'knn_10.quantile')::float as alt_p10_quantile
, json_extract_path_text(ETP_OUTPUT, 'Percentiles."15"')::float as final_p15
, json_extract_path_text(QUANTILES, 'knn_15')::float as orig_p15
, json_extract_path_text(ALT_QUANTILES, 'knn_15.value')::float as alt_p15
, json_extract_path_text(ALT_QUANTILES, 'knn_15.quantile')::float as alt_p15_quantile
, json_extract_path_text(ETP_OUTPUT, 'Percentiles."20"')::float as final_p20
, json_extract_path_text(QUANTILES, 'knn_20')::float as orig_p20
, json_extract_path_text(ALT_QUANTILES, 'knn_20.value')::float as alt_p20
, json_extract_path_text(ALT_QUANTILES, 'knn_20.quantile')::float as alt_p20_quantile
, json_extract_path_text(ETP_OUTPUT, 'Percentiles."25"')::float as final_p25
, json_extract_path_text(QUANTILES, 'knn_25')::float as orig_p25
, json_extract_path_text(ALT_QUANTILES, 'knn_25.value')::float as alt_p25
, json_extract_path_text(ALT_QUANTILES, 'knn_25.quantile')::float as alt_p25_quantile
, json_extract_path_text(ETP_OUTPUT, 'Percentiles."30"')::float as final_p30
, json_extract_path_text(QUANTILES, 'knn_30')::float as orig_p30
, json_extract_path_text(ALT_QUANTILES, 'knn_30.value')::float as alt_p30
, json_extract_path_text(ALT_QUANTILES, 'knn_30.quantile')::float as alt_p30_quantile
, json_extract_path_text(ETP_OUTPUT, 'Percentiles."35"')::float as final_p35
, json_extract_path_text(QUANTILES, 'knn_35')::float as orig_p35
, json_extract_path_text(ALT_QUANTILES, 'knn_35.value')::float as alt_p35
, json_extract_path_text(ALT_QUANTILES, 'knn_35.quantile')::float as alt_p35_quantile
, json_extract_path_text(ETP_OUTPUT, 'Percentiles."40"')::float as final_p40
, json_extract_path_text(QUANTILES, 'knn_40')::float as orig_p40
, json_extract_path_text(ALT_QUANTILES, 'knn_40.value')::float as alt_p40
, json_extract_path_text(ALT_QUANTILES, 'knn_40.quantile')::float as alt_p40_quantile
, json_extract_path_text(ETP_OUTPUT, 'Percentiles."45"')::float as final_p45
, json_extract_path_text(QUANTILES, 'knn_45')::float as orig_p45
, json_extract_path_text(ALT_QUANTILES, 'knn_45.value')::float as alt_p45
, json_extract_path_text(ALT_QUANTILES, 'knn_45.quantile')::float as alt_p45_quantile
, json_extract_path_text(ETP_OUTPUT, 'Percentiles."50"')::float as final_p50
, json_extract_path_text(QUANTILES, 'knn_50')::float as orig_p50
, json_extract_path_text(ALT_QUANTILES, 'knn_50.value')::float as alt_p50
, json_extract_path_text(ALT_QUANTILES, 'knn_50.quantile')::float as alt_p50_quantile
, json_extract_path_text(ETP_OUTPUT, 'Percentiles."55"')::float as final_p55
, json_extract_path_text(QUANTILES, 'knn_55')::float as orig_p55
, json_extract_path_text(ALT_QUANTILES, 'knn_55.value')::float as alt_p55
, json_extract_path_text(ALT_QUANTILES, 'knn_55.quantile')::float as alt_p55_quantile
, json_extract_path_text(ETP_OUTPUT, 'Percentiles."60"')::float as final_p60
, json_extract_path_text(QUANTILES, 'knn_60')::float as orig_p60
, json_extract_path_text(ALT_QUANTILES, 'knn_60.value')::float as alt_p60
, json_extract_path_text(ALT_QUANTILES, 'knn_60.quantile')::float as alt_p60_quantile
, json_extract_path_text(ETP_OUTPUT, 'Percentiles."65"')::float as final_p65
, json_extract_path_text(QUANTILES, 'knn_65')::float as orig_p65
, json_extract_path_text(ALT_QUANTILES, 'knn_65.value')::float as alt_p65
, json_extract_path_text(ALT_QUANTILES, 'knn_65.quantile')::float as alt_p65_quantile
, json_extract_path_text(ETP_OUTPUT, 'Percentiles."70"')::float as final_p70
, json_extract_path_text(QUANTILES, 'knn_70')::float as orig_p70
, json_extract_path_text(ALT_QUANTILES, 'knn_70.value')::float as alt_p70
, json_extract_path_text(ALT_QUANTILES, 'knn_70.quantile')::float as alt_p70_quantile
, json_extract_path_text(ETP_OUTPUT, 'Percentiles."75"')::float as final_p75
, json_extract_path_text(QUANTILES, 'knn_75')::float as orig_p75
, json_extract_path_text(ALT_QUANTILES, 'knn_75.value')::float as alt_p75
, json_extract_path_text(ALT_QUANTILES, 'knn_75.quantile')::float as alt_p75_quantile
, json_extract_path_text(ETP_OUTPUT, 'Percentiles."80"')::float as final_p80
, json_extract_path_text(QUANTILES, 'knn_80')::float as orig_p80
, json_extract_path_text(ALT_QUANTILES, 'knn_80.value')::float as alt_p80
, json_extract_path_text(ALT_QUANTILES, 'knn_80.quantile')::float as alt_p80_quantile
, json_extract_path_text(ETP_OUTPUT, 'Percentiles."85"')::float as final_p85
, json_extract_path_text(QUANTILES, 'knn_85')::float as orig_p85
, json_extract_path_text(ALT_QUANTILES, 'knn_85.value')::float as alt_p85
, json_extract_path_text(ALT_QUANTILES, 'knn_85.quantile')::float as alt_p85_quantile
, json_extract_path_text(ETP_OUTPUT, 'Percentiles."90"')::float as final_p90
, json_extract_path_text(QUANTILES, 'knn_90')::float as orig_p90
, json_extract_path_text(ALT_QUANTILES, 'knn_90.value')::float as alt_p90
, json_extract_path_text(ALT_QUANTILES, 'knn_90.quantile')::float as alt_p90_quantile
, json_extract_path_text(ETP_OUTPUT, 'Percentiles."95"')::float as final_p95
, json_extract_path_text(QUANTILES, 'knn_95')::float as orig_p95
, json_extract_path_text(ALT_QUANTILES, 'knn_95.value')::float as alt_p95
, json_extract_path_text(ALT_QUANTILES, 'knn_95.quantile')::float as alt_p95_quantile,
 case when  lc.carrier_shipment_charges_total <= final_p5  then 1 else 0 end as dqt_BeatP05,
 case when  lc.carrier_shipment_charges_total <= final_p10 then 1 else 0 end as dqt_BeatP10,
 case when  lc.carrier_shipment_charges_total <= final_p15 then 1 else 0 end as dqt_BeatP15,
 case when  lc.carrier_shipment_charges_total <= final_p20 then 1 else 0 end as dqt_BeatP20,
 case when  lc.carrier_shipment_charges_total <= final_p25 then 1 else 0 end as dqt_BeatP25,
 case when  lc.carrier_shipment_charges_total <= final_p30 then 1 else 0 end as dqt_BeatP30,
 case when  lc.carrier_shipment_charges_total <= final_p35 then 1 else 0 end as dqt_BeatP35,
 case when  lc.carrier_shipment_charges_total <= final_p40 then 1 else 0 end as dqt_BeatP40,
 case when  lc.carrier_shipment_charges_total <= final_p45 then 1 else 0 end as dqt_BeatP45,
 case when  lc.carrier_shipment_charges_total <= final_p50 then 1 else 0 end as dqt_BeatP50,
 case when  lc.carrier_shipment_charges_total <= final_p55 then 1 else 0 end as dqt_BeatP55,
 case when  lc.carrier_shipment_charges_total <= final_p60 then 1 else 0 end as dqt_BeatP60,
 case when  lc.carrier_shipment_charges_total <= final_p65 then 1 else 0 end as dqt_BeatP65,
 case when  lc.carrier_shipment_charges_total <= final_p70 then 1 else 0 end as dqt_BeatP70,
 case when  lc.carrier_shipment_charges_total <= final_p75 then 1 else 0 end as dqt_BeatP75,
 case when  lc.carrier_shipment_charges_total <= final_p80 then 1 else 0 end as dqt_BeatP80,
 case when  lc.carrier_shipment_charges_total <= final_p85 then 1 else 0 end as dqt_BeatP85,
 case when  lc.carrier_shipment_charges_total <= final_p90 then 1 else 0 end as dqt_BeatP90,
 case when  lc.carrier_shipment_charges_total <= final_p95 then 1 else 0 end as dqt_BeatP95,
 case when  lc.carrier_shipment_charges_total <= orig_p5  then 1 else 0 end as BeatP05,
 case when  lc.carrier_shipment_charges_total <= orig_p10 then 1 else 0 end as BeatP10,
 case when  lc.carrier_shipment_charges_total <= orig_p15 then 1 else 0 end as BeatP15,
 case when  lc.carrier_shipment_charges_total <= orig_p20 then 1 else 0 end as BeatP20,
 case when  lc.carrier_shipment_charges_total <= orig_p25 then 1 else 0 end as BeatP25,
 case when  lc.carrier_shipment_charges_total <= orig_p30 then 1 else 0 end as BeatP30,
 case when  lc.carrier_shipment_charges_total <= orig_p35 then 1 else 0 end as BeatP35,
 case when  lc.carrier_shipment_charges_total <= orig_p40 then 1 else 0 end as BeatP40,
 case when  lc.carrier_shipment_charges_total <= orig_p45 then 1 else 0 end as BeatP45,
 case when  lc.carrier_shipment_charges_total <= orig_p50 then 1 else 0 end as BeatP50,
 case when  lc.carrier_shipment_charges_total <= orig_p55 then 1 else 0 end as BeatP55,
 case when  lc.carrier_shipment_charges_total <= orig_p60 then 1 else 0 end as BeatP60,
 case when  lc.carrier_shipment_charges_total <= orig_p65 then 1 else 0 end as BeatP65,
 case when  lc.carrier_shipment_charges_total <= orig_p70 then 1 else 0 end as BeatP70,
 case when  lc.carrier_shipment_charges_total <= orig_p75 then 1 else 0 end as BeatP75,
 case when  lc.carrier_shipment_charges_total <= orig_p80 then 1 else 0 end as BeatP80,
 case when  lc.carrier_shipment_charges_total <= orig_p85 then 1 else 0 end as BeatP85,
 case when  lc.carrier_shipment_charges_total <= orig_p90 then 1 else 0 end as BeatP90,
 case when  lc.carrier_shipment_charges_total <= orig_p95 then 1 else 0 end as BeatP95 
, ROW_NUMBER() OVER (
        PARTITION BY etp.LoadNumber 
        ORDER BY event_timestamp_utc DESC
    ) AS rown
from events_raw.etp.etp_model_logs as etp
left join core_data.core.loads as lc on etp.loadnumber = lc.loadnumber
where event_timestamp_utc >= '2024-05-01'
)
where rown = 1
),
is_bounced as(
select is_bad_bounce, loadnumber
from (
select *, row_number() over(partition by LoadNumber order by bounce_date_time desc) as rown from core_data.core.bounces) 
where rown = 1
),
etp_percentiles as(
select
    is_etp_eligible, loadnumber
    from CORE_DATA.COMPONENTS.LOD__ETP_PERCENTILES
),
etp_lttc as ( --Data from lod__loadtargettruckcost
  select
  *
  , row_number() over(partition by etp.LoadNumber order by etp.CreatedOn desc) as rown
  from dapl_raw.accelerateprod.lod__loadtargettruckcost as etp
  where etp.CreatedOn > '2024-05-12' --Launch date
),
etp_letp as ( --Data from lod__loadelitetruckpurchasing
  select
  *
  , row_number() over(partition by etp.LoadNumber order by etp.CreatedOn desc) as rown
  from dapl_raw.accelerateprod.lod__loadelitetruckpurchasing as etp
  where etp.createdon > '2024-05-12' --Launch date
),
etp as (
    select
    lb.loadnumber
    , iff(etp_lttc.loadnumber is null, etp_letp.testgroup, etp_lttc.testgroup) as testgroup
    from core_data.components.lod__base as lb
    left join etp_lttc as etp_lttc on etp_lttc.loadnumber = lb.loadnumber and etp_lttc.rown=1
    left join etp_letp as etp_letp on etp_letp.loadnumber = lb.loadnumber and etp_letp.rown=1
    where (etp_lttc.rown=1 or etp_letp.rown=1) --Only loads in etp_lttc or etp_letp
),
car_accessorial_costs as(
select
    lb.LoadNumber as loadnumber,
    sum(lcc.TotalCost) as accessorial_cost
    from dapl_raw.accelerateprod.lod__loadboard_base lb
    left join dapl_raw.accelerateprod.lod__loadcarriercost lcc on lb.loadcarrierid = lcc.loadcarrierid
    left join dapl_raw.accelerateprod.lod__loadchargetype lct on lcc.chargetypeid = lct.chargetypeid
    where  lct.ChargeType != 'Extra Stop' and lct.ChargeTypeGrouping = 'Accessorial'
    group by lb.LoadNumber
),
final as (
select p.*, COALESCE(ib.is_bad_bounce, 0) as is_bad_bounce, etpp.is_etp_eligible, e.testgroup,
case when car.accessorial_cost is null then p.TotalCosts else p.TotalCosts - car.accessorial_cost end as TotalCosts_less_accessorials
from percentiles as p
left join is_bounced as ib on p.loadnumber = ib.loadnumber
left join etp_percentiles as etpp on etpp.loadnumber = p.loadnumber
left join etp as e on p.loadnumber = e.loadnumber
left join car_accessorial_costs as car on car.loadnumber = p.loadnumber
)
select *
from final
