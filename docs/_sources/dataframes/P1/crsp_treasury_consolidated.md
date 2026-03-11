# Dataframe: `P1:crsp_treasury_consolidated` - CRSP Treasury Consolidated

# crsp_treasury_consolidated

Placeholder.



## DataFrame Glimpse

```
Rows: 145747
Columns: 22
$ kytreasno                  <f64> 206021.0
$ kycrspid                   <str> '20100408.400000'
$ tcusip                     <str> '912795U3'
$ mcaldt            <datetime[ns]> 2009-11-30 00:00:00
$ tdatdt            <datetime[ns]> 2009-04-09 00:00:00
$ tmatdt            <datetime[ns]> 2010-04-08 00:00:00
$ tfcaldt                    <i64> 0
$ tfcpdt            <datetime[ns]> None
$ tmbid                      <f64> 99.965958333334
$ tmask                      <f64> 99.973125
$ tmaccint                   <f64> 0.0
$ tmyld                      <f64> 2.3614707616895e-06
$ price                      <f64> 99.969541666667
$ tcouprt                    <f64> 0.0
$ itype                      <f64> 4.0
$ original_maturity          <f64> 1.0
$ iflwr                      <f64> 1.0
$ years_to_maturity          <f64> 0.0
$ tmduratn                   <f64> 129.0
$ tmretnua                   <f64> 0.00031783665391802
$ days_to_maturity           <i64> 129
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
| Data available up to (min)     | None                                                             |
| Data available up to (max)     | None                                                             |
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
| Date of Last Code Update        | 2026-03-11 00:04:57           |
| OS Compatibility                |  |
| Linked Dataframes               |  [P1:crsp_treasury_daily_prices](../dataframes/P1/crsp_treasury_daily_prices.md)<br>  [P1:crsp_treasury_issue_info](../dataframes/P1/crsp_treasury_issue_info.md)<br>  [P1:crsp_treasury_consolidated](../dataframes/P1/crsp_treasury_consolidated.md)<br>  [P1:fed_yield_curve_all](../dataframes/P1/fed_yield_curve_all.md)<br>  [P1:fed_yield_curve](../dataframes/P1/fed_yield_curve.md)<br>  |


