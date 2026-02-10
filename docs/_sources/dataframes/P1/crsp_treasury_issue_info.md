# Dataframe: `P1:crsp_treasury_issue_info` - CRSP Treasury Issue Info

# crsp_treasury_issue_info

Placeholder.



## DataFrame Glimpse

```
Rows: 2264
Columns: 8
$ kytreasno                  <f64> 208504.0
$ kycrspid                   <str> '20321231.203880'
$ tcusip                     <str> '91282CPQ'
$ tdatdt            <datetime[ns]> 2025-12-31 00:00:00
$ tmatdt            <datetime[ns]> 2032-12-31 00:00:00
$ tcouprt                    <f64> 3.875
$ itype                      <f64> 2.0
$ original_maturity          <f64> 7.0


```

## Dataframe Manifest

| Dataframe Name                 | CRSP Treasury Issue Info                                                   |
|--------------------------------|--------------------------------------------------------------------------------------|
| Dataframe ID                   | [crsp_treasury_issue_info](../dataframes/P1/crsp_treasury_issue_info.md)                                       |
| Data Sources                   | CRSP Treasury Database                                        |
| Data Providers                 | CRSP, WRDS                                      |
| Links to Providers             |                              |
| Topic Tags                     |                                           |
| Type of Data Access            |                                   |
| How is data pulled?            | Pulled via WRDS SQL query and stored as issue-level metadata                                                    |
| Data available up to (min)     | 2025-12-31 00:00:00                                                             |
| Data available up to (max)     | 2025-12-31 00:00:00                                                             |
| Dataframe Path                 | /Users/phoebefingold/FINM_Repo/FINM_32900/p14_gurkaynak_sack_wright_2007/_data/TFZ_INFO.parquet                                                   |


**Linked Charts:**

- None


## Pipeline Manifest

| Pipeline Name                   | p14_us_treasury_yield_curve_construction_and_model_comparison                       |
|---------------------------------|--------------------------------------------------------|
| Pipeline ID                     | [P1](../index.md)              |
| Lead Pipeline Developer         | Phoebe Fingold, Annie Reynolds             |
| Contributors                    | Phoebe Fingold, Annie Reynolds           |
| Git Repo URL                    |                         |
| Pipeline Web Page               | <a href="file:///Users/phoebefingold/FINM_Repo/FINM_32900/p14_gurkaynak_sack_wright_2007/docs/index.html">Pipeline Web Page      |
| Date of Last Code Update        | 2026-02-09 22:42:02           |
| OS Compatibility                |  |
| Linked Dataframes               |  [P1:crsp_treasury_daily_prices](../dataframes/P1/crsp_treasury_daily_prices.md)<br>  [P1:crsp_treasury_issue_info](../dataframes/P1/crsp_treasury_issue_info.md)<br>  [P1:crsp_treasury_consolidated](../dataframes/P1/crsp_treasury_consolidated.md)<br>  [P1:fed_yield_curve_all](../dataframes/P1/fed_yield_curve_all.md)<br>  [P1:fed_yield_curve](../dataframes/P1/fed_yield_curve.md)<br>  |


