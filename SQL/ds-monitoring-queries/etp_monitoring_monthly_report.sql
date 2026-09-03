/*
Purpose: ETP monitoring monthly report extract.
Owner: Steve
Params: none
Consumers: Tableau; notebooks/ds-monitoring.ipynb
Links:
  https://prod-useast-b.online.tableau.com/#/site/arrivelogistics/views/ETPMonitoringMonthlyReport/ETPMonitoringDashboardVisualizationsPt_ALLDaily
  https://prod-useast-b.online.tableau.com/#/site/arrivelogistics/views/ETPMonitoringMonthlyReport/ETPMonitoringDashboard10th
*/

with
etp_lttc as ( --Data from lod__loadtargettruckcost
  select
  *
  , row_number() over(partition by etp.LoadNumber order by etp.CreatedOn desc) as rown
  from dapl_raw.accelerateprod.lod__loadtargettruckcost as etp
  where etp.CreatedOn > '2022-09-18' --Launch date
),
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
    , iff(etp_lttc.loadnumber is null, etp_letp.targetcost, etp_lttc.FinalValue) as etp
    , iff(etp_lttc.loadnumber is null, etp_letp.testgroup, etp_lttc.testgroup) as testgroup
    , iff(etp_lttc.loadnumber is null, etp_letp.ineligibilityreason, etp_lttc.ineligibilityreason) as ineligibilityreason
    , iff(etp_lttc.loadnumber is null, etp_letp.istargeteligible, etp_lttc.ISTARGETCOSTELIGIBLE) as istargeteligible
    from core_data.components.lod__base as lb
    left join etp_lttc as etp_lttc on etp_lttc.loadnumber = lb.loadnumber and etp_lttc.rown=1
    left join etp_letp as etp_letp on etp_letp.loadnumber = lb.loadnumber and etp_letp.rown=1
    where (etp_lttc.rown=1 or etp_letp.rown=1) --Only loads in etp_lttc or etp_letp
),
booked_loads as (
    select
    l.loadnumber,
    l.priced_to_make as pricedtomake,
    l.booked_on_cst,
    vmm.booking_velocity,
    l.Customer_charges_total as TotalCharges,
    l.Carrier_charges_total as TotalCosts,
    l.Customer_charges_total - l.Carrier_charges_total as TotalSpread,
    l.is_loss_load,
    l.market_rate,
    l.market_comp,
    CASE
    WHEN pricedtomake between 0 and 250 THEN '0 to 250'
    WHEN pricedtomake > 250 THEN 'More than 250'
    WHEN pricedtomake < 0 THEN 'Less than 0'
    END as PricedToMakeBucket,
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
etp_percentiles as(
select
    *
    from CORE_DATA.COMPONENTS.LOD__ETP_PERCENTILES
),
is_bounced as(
select is_bad_bounce, loadnumber
from (
select *, row_number() over(partition by LoadNumber order by bounce_date_time desc) as rown from core_data.core.bounces) 
where rown = 1
),
final as (
    select
    bl.LoadNumber,
    etpp.IS_ETP_ELIGIBLE,
    etpp.IS_ETP_ELIGIBLE_FINANCE,
    etpp.P05,
    etpp.P10,
    etpp.P15,
    etpp.P20,
    etpp.P25,
    etpp.P30,
    etpp.P35,
    etpp.P40,
    etpp.P45,
    etpp.P50,
    etpp.P55,
    etpp.P60,
    etpp.P65,
    etpp.P70,
    etpp.P75,
    etpp.P80,
    etpp.P85,
    etpp.P90,
    etpp.P95,
    bl.pricedtomake,
    bl.PricedToMakeBucket,
    COALESCE(ib.is_bad_bounce, 0) as is_bad_bounce,
    bl.is_booked_load,
    bl.booked_on_cst,
    iff(btl.is_booked_tl is null, 0, btl.is_booked_tl) as is_booked_tl,
    iff(bel.booked_etp_eligible is null, 0, bel.booked_etp_eligible) as booked_etp_eligible,
    iff(etp.ETP is not null, 1, 0) as went_through_etp,
    iff(etp.ETP > 0, 1, 0) as has_etp,
    iff(etp.ETP is not null and etp.istargeteligible = TRUE, 1, 0) as has_etp_on_screen,
    bl.TotalCharges,
    bl.TotalCosts,
    case when acc.accessorial_cost is null then bl.TotalCosts else bl.TotalCosts - acc.accessorial_cost end as TotalCosts_less_accessorials,
    bl.TotalSpread,
    iff(bl.TotalSpread <0,1,0) as is_loss_load,
    bl.market_rate,
    bl.market_comp as cost_less_marlet_rt,
    etp.etp,
    (iff(etp.etp = 0, NULL, etp.etp)-bl.market_rate) as target_cost_less_market_rt,
    ps.is_prebook_eligible,
    ps.is_prebooked,
    ps.booked_to_cutoff_hours,
    bl.booking_velocity,
    etp.testgroup,
    etp.ineligibilityreason,
    etp.istargeteligible
    from  booked_loads as bl
    left join prebook_stats ps on bl.loadnumber = ps.loadnumber
    left join booked_truck_load as btl on btl.loadnumber=bl.loadnumber
    left join booked_etp_eligible as bel on bel.loadnumber=bl.loadnumber
    left join etp on etp.loadnumber=bl.loadnumber
    left join car_accessorial_costs as acc on acc.loadnumber=bl.loadnumber
    left join etp_percentiles as etpp on etpp.loadnumber=bl.loadnumber
    left join is_bounced as ib on ib.loadnumber = bl.loadnumber
)
select * from final