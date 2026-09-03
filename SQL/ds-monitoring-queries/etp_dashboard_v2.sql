/*
Purpose: ETP Dashboard v2 financial metrics extract.
Owner: Steve
Params: none
Consumers: Tableau; notebooks/ds-monitoring.ipynb
Link: https://prod-useast-b.online.tableau.com/#/site/arrivelogistics/views/ETPDashboardv2/FinancialMetrics2
*/

with
--etp_lttc as ( --Data from lod__loadtargettruckcost
  --select
  --*
 -- , row_number() over(partition by etp.LoadNumber order by etp.CreatedOn desc) as rown
--  from dapl_raw.accelerateprod.lod__loadtargettruckcost as etp
--  where etp.CreatedOn > '2022-09-18' --Launch date
--),
etp_letp as ( --Data from lod__loadelitetruckpurchasing
  select
  *
  , row_number() over(partition by etp.LoadNumber order by etp.CreatedOn desc) as rown
  from dapl_raw.accelerateprod.lod__loadelitetruckpurchasing as etp
  where etp.createdon > '2022-09-18' --Launch date
),
etp as (
    select
    lb.loadnumber
    , json_extract_path_text(etp_letp.percentiles, '"5"')::float as P5 --P25
    , json_extract_path_text(etp_letp.percentiles, '"10"')::float as P10 --P25
    , json_extract_path_text(etp_letp.percentiles, '"15"')::float as P15 --P25
    , json_extract_path_text(etp_letp.percentiles, '"20"')::float as P20 --P25
    , json_extract_path_text(etp_letp.percentiles, '"25"')::float as P25 --P25
    , json_extract_path_text(etp_letp.percentiles, '"30"')::float as P30 --P25
    , json_extract_path_text(etp_letp.percentiles, '"35"')::float as P35 --P25
    , json_extract_path_text(etp_letp.percentiles, '"40"')::float as P40 --P25
    , json_extract_path_text(etp_letp.percentiles, '"45"')::float as P45 --P25
    , json_extract_path_text(etp_letp.percentiles, '"50"')::float as P50 --P25
    , json_extract_path_text(etp_letp.percentiles, '"55"')::float as P55 --P25
    , json_extract_path_text(etp_letp.percentiles, '"60"')::float as P60 --P25
    , json_extract_path_text(etp_letp.percentiles, '"65"')::float as P65 --P25
    , json_extract_path_text(etp_letp.percentiles, '"70"')::float as P70 --P25
    , json_extract_path_text(etp_letp.percentiles, '"75"')::float as P75 --P25
    , json_extract_path_text(etp_letp.percentiles, '"80"')::float as P80 --P25
    , json_extract_path_text(etp_letp.percentiles, '"85"')::float as P85 --P25
    , json_extract_path_text(etp_letp.percentiles, '"90"')::float as P90 --P25
    , json_extract_path_text(etp_letp.percentiles, '"95"')::float as P95 --P25
    --, iff(etp_lttc.loadnumber is null, etp_letp.targetcost, etp_lttc.FinalValue) as etp
    , etp_letp.testgroup as testgroup
    , etp_letp.ineligibilityreason as ineligibilityreason
    , etp_letp.istargeteligible as istargeteligible
    ,  case when etp_letp.istargeteligible = 'TRUE' then 'test' when etp_letp.ineligibilityreason = 'control' then 'control' end as test_group_fixed
    from core_data.components.lod__base as lb
    --left join etp_lttc as etp_lttc on etp_lttc.loadnumber = lb.loadnumber and etp_lttc.rown=1
    left join etp_letp as etp_letp on etp_letp.loadnumber = lb.loadnumber and etp_letp.rown=1
    where (etp_letp.rown=1) --Only loads in etp_lttc or etp_letp
  --where (etp_lttc.rown=1 or etp_letp.rown=1) --Only loads in etp_lttc or etp_letp
),
booked_loads as (
    select
    l.loadnumber,
    l.priced_to_make as pricedtomake,
    l.booked_on_cst,
    l.DELIVERED_ON_ACTUAL_LOCAL,
    l.PICKED_UP_ACTUAL_ON_LOCAL,
    vmm.booking_velocity,
    l.Customer_charges_total as TotalCharges,
    l.Carrier_charges_total as TotalCosts,
    l.Customer_Shipment_charges_total,
    l.Carrier_shipment_charges_total,
    l.Customer_charges_total - l.Carrier_charges_total as TotalSpread,
    l.is_loss_load,
    l.market_rate,
    l.market_comp,
    l.origin_market_area_id,
    l.dest_market_area_id,
    1 as is_booked_load
    from core_data.core.loads as l
    left join core_data.components.lod__status_mapping as sm on l.orderstatus=sm.order_status
    left join core_data.components.lod__velocity vmm on l.loadnumber = vmm.loadnumber
    where sm.is_covered=1 --Booked+
    and l.Customer_Shipment_charges_total>250
    and l.Carrier_shipment_charges_total >250
    and l.is_tonu=False
),
prebook_stats as (
SELECT
        LOADNUMBER,
        PU_LATEST_AT_SNAPSHOT,
        IS_PREBOOK_ELIGIBLE,
	    booked_to_cutoff_hours,
        IS_PREBOOKED
FROM
       (SELECT *, RANK() OVER (PARTITION BY LOADNUMBER ORDER BY SNAPSHOT_CREATEDON_CST desc) as row_num  FROM core_data.components.lod__prebook ) a
WHERE
     row_num = 1
),
booked_truck_load as (
    select
    l.loadnumber
  , 1 as is_booked_tl
    from core_data.core.loads as l
    left join dapl_raw.accelerateprod.lod__loadboard_base lb on l.loadnumber = lb.loadnumber
    left join core_data.components.lod__status_mapping as sm on l.orderstatus=sm.order_status
    where sm.is_covered=1 --Booked+
-- Truckload Loads
    and lb.TotalCharges >250
    and lower(lb.orderstatus) not in ('bounced','void','available','assigned','hold')
    and lower(lb.loadtypedescription) not in ('imdl','ltl','drayage','power only','partial','air')
),
booked_etp_eligible as (
    select
    l.loadnumber,
    1 as booked_etp_eligible
    from core_data.core.loads as l
    left join dapl_raw.accelerateprod.lod__loadboard_base lb on l.loadnumber = lb.loadnumber
    left join dapl_raw.accelerateprod.ops__Address o on lb.PickupEarlyFacilityId = o.FacilityId
    left join dapl_raw.accelerateprod.ops__Address d on lb.DeliveryLateFacilityId = d.FacilityId
    left join core_data.components.lod__status_mapping as sm on l.orderstatus=sm.order_status
    where sm.is_covered=1 --Booked+
    --ETP eligible loads
    and lower(lb.orderstatus) not in ('bounced','void','available','assigned','hold')
    and lb.Miles>0
    and lb.TotalCharges >250
    and lb.NumberOfPickups<2
    and lb.NumberOfDeliveries<2
    and lb.equipmenttype in ('V','VR','R')
    and lb.LoadTypeDescription in ('REEFER','DRY')
    and (lb.DropTrailer = 0 or lb.DropTrailer is NULL)
    and o.country  in ('US','USA','United States')
    and d.country  in ('US','USA','United States')
    and (l.is_tonu=0 or is_tonu is null)
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
is_bounced as(
select is_bad_bounce, loadnumber
from (
select *, row_number() over(partition by LoadNumber order by bounce_date_time desc) as rown from core_data.core.bounces) 
where rown = 1
),
etp_percentiles as(
select loadnumber, IS_ETP_ELIGIBLE, IS_ETP_ELIGIBLE_FINANCE
from CORE_DATA.COMPONENTS.LOD__ETP_PERCENTILES
),
twic as (
    select
	ls.loadnumber,
	from DAPL_RAW_QA.ACCELERATEPROD.LOD__LOADSTOP as ls
	where ls.twicneeded=1 --twic in any pickup and delivery
	group by ls.loadnumber
),
bonded as (
    select
	less.loadnumber,
	from DAPL_RAW_QA.ACCELERATEPROD.LOD__LOADEQUIPMENTSPECIALSERVICE as less
	where less.LoadSpecialServiceId in (95) --bonded
	group by less.loadnumber
),
mde as (
select *
from (
select loadnumber, createdon as created_on_mde, MARKETDISRUPTIONEVENTLISTID, row_number() over(partition by loadnumber order by modifiedon desc) as rown
from DAPL_RAW.ACCELERATEPROD.LOD__DISPLAYEDTARGET
)
where rown = 1
),
lane_destroyer as (
select *
-- , case when snowflakeupdatedon = (select max(snowflakeupdatedon) from data_science_staged_data.etp.etp_lane_destroyer_lanes_history)
-- then 'current' else 'legacy' end as tag
from data_science_staged_data.etp.etp_lane_destroyer_lanes_history
),
displayed_targets as (
SELECT 
    loadnumber,
    MAX(CASE WHEN displayid = 1 THEN etpvalue END) AS ETP1,
    MAX(CASE WHEN displayid = 2 THEN etpvalue END) AS ETP2,
    MAX(CASE WHEN displayid = 3 THEN etpvalue END) AS ETP3,
    MAX(CASE WHEN displayid = 4 THEN etpvalue END) AS ETP4
FROM 
    (
        SELECT 
            loadnumber,
            displayid,
            etpvalue,
            ROW_NUMBER() OVER (PARTITION BY loadnumber ORDER BY modifiedon DESC) AS rn
        FROM 
            DAPL_RAW.ACCELERATEPROD.LOD__DISPLAYEDTARGET
    ) AS RankedRows
WHERE 
    rn <= 4
GROUP BY 
    loadnumber
),
final as (
    select
    bl.LoadNumber,
    bl.is_booked_load,
    bl.booked_on_cst,
    bl.pricedtomake,
    CASE WHEN tw.loadnumber IS NULL THEN 0 ELSE 1 END AS is_twic,
    CASE WHEN bd.loadnumber IS NULL THEN 0 ELSE 1 END AS is_bonded,
    bl.DELIVERED_ON_ACTUAL_LOCAL,
    bl.PICKED_UP_ACTUAL_ON_LOCAL,
    etpp.is_etp_eligible,
    etpp.is_etp_eligible_finance,
    ps.PU_LATEST_AT_SNAPSHOT,
    COALESCE(ib.is_bad_bounce, 0) as is_bad_bounce,
    iff(btl.is_booked_tl is null, 0, btl.is_booked_tl) as is_booked_tl,
    iff(bel.booked_etp_eligible is null, 0, bel.booked_etp_eligible) as booked_etp_eligible,
    iff(dp.ETP1 is not null, 1, 0) as went_through_etp,
    iff(dp.ETP1 > 0, 1, 0) as has_etp,
    iff(dp.ETP1 is not null and etp.istargeteligible = TRUE, 1, 0) as has_etp_on_screen,
    bl.TotalCharges,
    bl.TotalCosts,    
    bl.Customer_Shipment_charges_total,
    bl.Carrier_shipment_charges_total,
    case when acc.accessorial_cost is null then bl.TotalCosts else bl.TotalCosts - acc.accessorial_cost end as TotalCosts_less_accessorials,
    bl.TotalSpread,
    iff(bl.TotalSpread <0,1,0) as is_loss_load,
    bl.market_rate,
    bl.market_comp as cost_less_marlet_rt,
    etp.* exclude (loadnumber),
    dp.* exclude (loadnumber, ETP1, ETP2),
    lttc.testgroup as lttc_testgroup,
    case when dp.ETP1 = 0 then null else ETP1 end as ETP1,
    dp.ETP2,
    (iff(dp.ETP1 = 0, NULL, dp.ETP1)-bl.market_rate) as target_cost_less_market_rt_1,
    (iff(dp.ETP2 = 0, NULL, dp.ETP2)-bl.market_rate) as target_cost_less_market_rt_2,
    (iff(dp.ETP3 = 0, NULL, dp.ETP3)-bl.market_rate) as target_cost_less_market_rt_3,
    (iff(dp.ETP4 = 0, NULL, dp.ETP4)-bl.market_rate) as target_cost_less_market_rt_4,
    ps.is_prebook_eligible,
    ps.is_prebooked,
    ps.booked_to_cutoff_hours,
    bl.booking_velocity as bookingvelocity,
    md.MARKETDISRUPTIONEVENTLISTID,
    md.created_on_mde,
    CASE WHEN MARKETDISRUPTIONEVENTLISTID is not null then 1 else 0 end as mde_ind,
    CASE WHEN ld.origin_market_area_id IS NOT NULL THEN 1 ELSE 0 END AS lane_flag,
    ld.test_group,
    ld.tag,
    mdlist.description AS mde_description,
    from  booked_loads as bl
    left join prebook_stats ps on bl.loadnumber = ps.loadnumber
    left join booked_truck_load as btl on btl.loadnumber=bl.loadnumber
    left join booked_etp_eligible as bel on bel.loadnumber=bl.loadnumber
    left join etp on etp.loadnumber=bl.loadnumber
    left join car_accessorial_costs as acc on acc.loadnumber=bl.loadnumber
    left join is_bounced as ib on ib.loadnumber=bl.loadnumber
    left join dapl_raw.accelerateprod.lod__loadtargettruckcost lttc on lttc.loadnumber =bl.loadnumber
    left join CORE_DATA.COMPONENTS.LOD__ETP_PERCENTILES as etpp on bl.loadnumber=etpp.loadnumber
    left join twic as tw on bl.loadnumber = tw.loadnumber
    left join bonded as bd on bl.loadnumber = bd.loadnumber
    left join mde as md on bl.loadnumber = md.loadnumber
    left join lane_destroyer as ld on
	bl.origin_market_area_id = ld.origin_market_area_id 
	AND bl.dest_market_area_id = ld.dest_market_area_id
	AND bl.booked_on_cst >= ld.effective_date
	AND (bl.booked_on_cst < ld.end_date OR ld.end_date is null)
    left join displayed_targets as dp on bl.loadnumber = dp.loadnumber
    LEFT JOIN core_data.components.lod__market_disruption_event_loads mdloads ON bl.loadnumber = mdloads.loadnumber
    LEFT JOIN core_data.components.lod__market_disruption_event_list mdlist
        ON mdlist.market_disruption_event_list_id = mdloads.market_disruption_event_list_id 
)
select * from final
where booked_on_cst >= '2023-08-30'
