# Data Dictionary

Data downloaded: 2026-03-17

## File 1: raw_monthly_data.csv

Monthly frequency. Daily series (exchange rate, policy rate, expectations) are
averaged to monthly. CPI-related series reported as monthly % variations by the
source are cumulated into price indices (base = 100 at first observation).

| Column | Description | Unit | Source | Series Code |
|--------|-------------|------|--------|-------------|
| IC_Br_BRL | IC-Br Commodity Index — General (BRL) | index | BCB_SGS | 27574 |
| IC_Br_USD | IC-Br Commodity Index — General (USD) | index | BCB_SGS | 29042 |
| IC_Br_Agro_BRL | IC-Br Commodity Index — Agriculture (BRL) | index | BCB_SGS | 27575 |
| IC_Br_Agro_USD | IC-Br Commodity Index — Agriculture (USD) | index | BCB_SGS | 29041 |
| IC_Br_Energy_BRL | IC-Br Commodity Index — Energy (BRL) | index | BCB_SGS | 27577 |
| IC_Br_Energy_USD | IC-Br Commodity Index — Energy (USD) | index | BCB_SGS | 29039 |
| IC_Br_Metal_BRL | IC-Br Commodity Index — Metals (BRL) | index | BCB_SGS | 27576 |
| IC_Br_Metal_USD | IC-Br Commodity Index — Metals (USD) | index | BCB_SGS | 29040 |
| Terms_of_Trade | Brazil Terms of Trade Index | index | BCB_SGS | 11753 |
| Exchange_Rate | Exchange Rate BRL/USD (buy, daily average) | brl_usd | BCB_SGS | 1 |
| CPI_Brazil | IPCA headline CPI (monthly % variation → index) | pct_mom | BCB_SGS | 433 |
| Core_IPCA_EX0 | Core IPCA EX0 (monthly % variation → index) | pct_mom | BCB_SGS | 11427 |
| Core_IPCA_EX3 | Core IPCA EX3 (monthly % variation → index) | pct_mom | BCB_SGS | 27839 |
| Core_IPCA_Trimmed_Smoothed | Core IPCA Smoothed Trimmed Mean (monthly % var → index) | pct_mom | BCB_SGS | 4466 |
| Core_IPCA_Double_Weight | Core IPCA Double Weighted (monthly % var → index) | pct_mom | BCB_SGS | 16122 |
| Core_IPCA_P55 | Core IPCA P55 (monthly % var → index) | pct_mom | BCB_SGS | 28750 |
| Brazil_Policy_Rate | Selic Target Rate (% p.a.) | pct_pa | BCB_SGS | 1178 |
| IPCA_12M_Expectation | IPCA 12-Month Ahead Median Expectation (% p.a.) | pct_pa | BCB_OLINDA | ExpectativasMercadoInflacao12Meses |
| US_Potential_GDP | US CBO Potential GDP (Billions of Chained 2017 Dollars) | index | FRED | GDPPOT |
| US_Real_GDP | US Real GDP (Billions of Chained 2017 Dollars) | index | FRED | GDPC1 |
| US_NAIRU | US Natural Rate of Unemployment (%) | pct_pa | FRED | NROU |
| US_CPI | US CPI All Urban Consumers (1982-84=100) | index | FRED | CPIAUCSL |
| US_Fed_Funds_Rate | US Federal Funds Effective Rate (% p.a.) | pct_pa | FRED | FEDFUNDS |
| Global_Oil_Price_WTI | Oil Price — Brent Crude (USD/barrel, monthly avg) | usd | FRED | POILBREUSDM |
| Global_Food_Price_Index | IMF Global Food Price Index (2016=100) | index | FRED | PFOODINDEXM |
| US_Output_Gap | US Output Gap (CBO-based, %) | pct_pa | COMPUTED | 100*(GDPC1-GDPPOT)/GDPPOT |
| CPI_FOOD | Food & Beverages CPI Component (index, from multiple table vintages) | index | SIDRA | t7060 |
| CPI_ENERGY | Energy CPI Component — Electricity + Fuels (index) | index | SIDRA | t7060 |
| CDS5_Brazil_bps | 5-Year CDS Spread | basis points | Bloomberg (LOCAL) | CDS_5yr_10yr.csv |
| CDS10_Brazil_bps | 10-Year CDS Spread | basis points | Bloomberg (LOCAL) | CDS_5yr_10yr.csv |
| Brazil_Inflation_Target | BCB Inflation Target Center (% p.a.) | pct_pa | LOCAL | Inflation_target.csv |

## File 2: consolidated_quarterly_data.csv

Quarterly frequency. Monthly/daily series are aggregated to quarterly using the
method indicated below (typically mean). CDS is last monthly value per quarter,
divided by 100. Inflation target is expanded from annual to quarterly.

| Column | Description | Unit | Source | Series Code | Aggregation |
|--------|-------------|------|--------|-------------|-------------|
| CDS5_Brazil | 5-Year CDS Spread (basis points, Bloomberg) | bps | LOCAL | CDS_5yr_10yr.csv | last |
| Brazil_Inflation_Target | BCB Inflation Target Center (% p.a.) | pct_pa | LOCAL | Inflation_target.csv | mean |
| Brazil_Inflation_Target_Lower | BCB Inflation Target Lower Bound (% p.a.) | pct_pa | LOCAL | Inflation_target.csv | mean |
| Brazil_Inflation_Target_Upper | BCB Inflation Target Upper Bound (% p.a.) | pct_pa | LOCAL | Inflation_target.csv | mean |
| US_Trend_Growth | HLW Trend Growth Rate (%) | pct_pa | NYFED | trend_growth | direct |
| US_Natural_Rate | Holston-Laubach-Williams Natural Rate of Interest (r*) | pct_pa | NYFED | rstar | direct |
| US_Output_Gap_HLW | HLW Output Gap (%) | pct_pa | NYFED | output_gap | direct |
| US_Potential_GDP | US CBO Potential GDP (Billions of Chained 2017 Dollars) | index | FRED | GDPPOT | direct |
| US_Real_GDP | US Real GDP (Billions of Chained 2017 Dollars) | index | FRED | GDPC1 | direct |
| US_NAIRU | US Natural Rate of Unemployment (%) | pct_pa | FRED | NROU | direct |
| US_CPI | US CPI All Urban Consumers (1982-84=100) | index | FRED | CPIAUCSL | mean |
| US_Fed_Funds_Rate | US Federal Funds Effective Rate (% p.a.) | pct_pa | FRED | FEDFUNDS | mean |
| Global_Oil_Price_WTI | Oil Price — Brent Crude (USD/barrel, monthly avg) | usd | FRED | POILBREUSDM | mean |
| Global_Food_Price_Index | IMF Global Food Price Index (2016=100) | index | FRED | PFOODINDEXM | mean |
| US_Output_Gap | US Output Gap (CBO-based, %) | pct_pa | COMPUTED | 100*(GDPC1-GDPPOT)/GDPPOT | direct |
| IPCA_12M_Expectation | IPCA 12-Month Ahead Median Expectation (% p.a.) | pct_pa | BCB_OLINDA | ExpectativasMercadoInflacao12Meses | mean |
| GDP_Brazil_Quarterly | Quarterly GDP — Chained Volume Index (1995=100) | index | SIDRA | t1620/v583 | direct |
| CPI_FOOD | Food & Beverages CPI Component (index, from multiple table vintages) | index | SIDRA | t7060 | mean |
| CPI_ENERGY | Energy CPI Component — Electricity + Fuels (index) | index | SIDRA | t7060 | mean |
| W_FOOD | Food Weight in CPI Basket (%) | pct_pa | SIDRA | t7060 | mean |
| W_ENERGY | Energy Weight in CPI Basket (%) | pct_pa | SIDRA | t7060 | mean |
| IC_Br_BRL | IC-Br Commodity Index — General (BRL) | index | BCB_SGS | 27574 | mean |
| IC_Br_USD | IC-Br Commodity Index — General (USD) | index | BCB_SGS | 29042 | mean |
| IC_Br_Agro_BRL | IC-Br Commodity Index — Agriculture (BRL) | index | BCB_SGS | 27575 | mean |
| IC_Br_Agro_USD | IC-Br Commodity Index — Agriculture (USD) | index | BCB_SGS | 29041 | mean |
| IC_Br_Energy_BRL | IC-Br Commodity Index — Energy (BRL) | index | BCB_SGS | 27577 | mean |
| IC_Br_Energy_USD | IC-Br Commodity Index — Energy (USD) | index | BCB_SGS | 29039 | mean |
| IC_Br_Metal_BRL | IC-Br Commodity Index — Metals (BRL) | index | BCB_SGS | 27576 | mean |
| IC_Br_Metal_USD | IC-Br Commodity Index — Metals (USD) | index | BCB_SGS | 29040 | mean |
| Terms_of_Trade | Brazil Terms of Trade Index | index | BCB_SGS | 11753 | mean |
| Exchange_Rate | Exchange Rate BRL/USD (buy, daily average) | brl_usd | BCB_SGS | 1 | mean |
| CPI_Brazil | IPCA headline CPI (monthly % variation → index) | pct_mom | BCB_SGS | 433 | mean |
| GDP_Brazil_SA | GDP Seasonally Adjusted (BCB monthly proxy) | index | BCB_SGS | 22109 | mean |
| IBC_Br | IBC-Br Economic Activity Index (SA) | index | BCB_SGS | 24363 | mean |
| Core_IPCA_EX0 | Core IPCA EX0 (monthly % variation → index) | pct_mom | BCB_SGS | 11427 | mean |
| Core_IPCA_EX3 | Core IPCA EX3 (monthly % variation → index) | pct_mom | BCB_SGS | 27839 | mean |
| Core_IPCA_Trimmed_Smoothed | Core IPCA Smoothed Trimmed Mean (monthly % var → index) | pct_mom | BCB_SGS | 4466 | mean |
| Core_IPCA_Double_Weight | Core IPCA Double Weighted (monthly % var → index) | pct_mom | BCB_SGS | 16122 | mean |
| Core_IPCA_P55 | Core IPCA P55 (monthly % var → index) | pct_mom | BCB_SGS | 28750 | mean |
| Brazil_Policy_Rate | Selic Target Rate (% p.a.) | pct_pa | BCB_SGS | 1178 | mean |
| Core_IPCA_Average | Average of BCB Core IPCA Measures (index) | index | COMPUTED | mean(EX0,EX3,Trimmed,DW,P55) | direct |

## Sources

| Code | Full Name | URL |
|------|-----------|-----|
| BCB_SGS | Banco Central do Brasil — Time Series Management System | https://api.bcb.gov.br |
| BCB_OLINDA | Banco Central do Brasil — Market Expectations (OData) | https://olinda.bcb.gov.br |
| SIDRA | IBGE — Sistema IBGE de Recuperacao Automatica | https://apisidra.ibge.gov.br |
| FRED | Federal Reserve Economic Data (St. Louis Fed) | https://fred.stlouisfed.org |
| NYFED | New York Fed — Holston-Laubach-Williams Estimates | https://www.newyorkfed.org/research/policy/rstar |
| LOCAL | Static files (Bloomberg CDS, BCB policy decisions) | — |
| COMPUTED | Derived in pipeline | — |
