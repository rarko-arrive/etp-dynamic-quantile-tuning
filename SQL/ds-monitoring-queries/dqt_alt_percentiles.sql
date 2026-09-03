/*
Purpose: Latest HISTORICAL_ETP_DYNAMIC_QUANTILES row per valid_date.
Owner: Steve
Params: none
Consumers: Tableau ETP Alt Percentiles; dqt.model.global_dqt; notebooks/ds-monitoring.ipynb
Link: https://prod-useast-b.online.tableau.com/#/site/arrivelogistics/views/ETPAltPercentiles/Dashboard1
*/

SELECT *
FROM (
SELECT *, ROW_NUMBER() OVER (
        PARTITION BY valid_date 
        ORDER BY SNOWFLAKEUPDATEDON DESC
    ) AS rown
    FROM DATA_SCIENCE.ETP_DYNAMIC_QUANTILE_TUNING.HISTORICAL_ETP_DYNAMIC_QUANTILES
    )
WHERE rown=1