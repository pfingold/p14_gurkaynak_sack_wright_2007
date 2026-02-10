# Dataframe: `P1:crsp_treasury_consolidated` - CRSP Treasury Consolidated

# crsp_treasury_consolidated

Placeholder.



## DataFrame Glimpse

```
Rows: 2530592
Columns: 20
$ kytreasno                  <f64> 200636.0
$ kycrspid                   <str> '19700215.104000'
$ tcusip                     <str> '912810AE'
$ caldt             <datetime[ns]> 1970-01-02 00:00:00
$ tdatdt            <datetime[ns]> 1965-01-15 00:00:00
$ tmatdt            <datetime[ns]> 1970-02-15 00:00:00
$ tfcaldt                    <i64> 0
$ tdbid                      <f64> 99.5
$ tdask                      <f64> 99.5625
$ tdaccint                   <f64> 1.5217391304348
$ tdyld                      <f64> 0.00021199520830037
$ price                      <f64> 101.0529891304348
$ tcouprt                    <f64> 4.0
$ itype                      <f64> 1.0
$ original_maturity          <f64> 5.0
$ years_to_maturity          <f64> 0.0
$ tdduratn                   <f64> 44.0
$ tdretnua                   <f64> 0.00083430893652524
$ days_to_maturity           <i64> 44
$ callable                  <bool> False


```

## Dataframe Manifest

| Dataframe Name                 | CRSP Treasury Consolidated                                                   |
|--------------------------------|--------------------------------------------------------------------------------------|
| Dataframe ID                   | [crsp_treasury_consolidated](../dataframes/P1/crsp_treasury_consolidated.md)                                       |
| Data Sources                   | CRSP Treasury Database                                        |
| Data Providers                 | WRDS                                      |
| Links to Providers             |                              |
| Topic Tags                     |                                           |
| Type of Data Access            |                                   |
| How is data pulled?            | Pulled via WRDS SQL query and merged in Python                                                    |
| Data available up to (min)     | N/A (large file)                                                             |
| Data available up to (max)     | N/A (large file)                                                             |
| Dataframe Path                 | /Users/phoebefingold/FINM_Repo/FINM_32900/p14_gurkaynak_sack_wright_2007/_data/TFZ_consolidated.parquet                                                   |


**Linked Charts:**


- [P1:crsp_treasury_sample_plot](../../charts/P1.crsp_treasury_sample_plot.md)



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


