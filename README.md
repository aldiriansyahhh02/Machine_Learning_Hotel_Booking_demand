<h1> Hotel Booking Cancellation Prediction with Machine Learning </h1>

## 1. Project Overview
This project analyzes hotel reservation data from a hotel in Portugal to extract insights, improve decision-making, and identify key trends. The primary focus is to **predict the probability of each booking being canceled (`is_canceled`)** so the hotel can anticipate empty rooms, take preventive action on high-risk guests, and understand which factors drive cancellations. About 37% of bookings in the data end in cancellation, which causes lost revenue and inaccurate occupancy, staffing, and inventory planning. Without a clear basis, overbooking also risks relocating guests who actually arrive.

Key Objectives:
- Objective 1: Analyze which booking characteristics (deposit type, market segment, guest origin, special requests, etc.) are related to cancellations
- Objective 2: Build a binary classification model (Random Forest) to predict cancellation probability, with target **F2-score ≥ 0.75** and **cancel-class recall ≥ 80%** on test data
- Objective 3: Reduce total prediction-error cost (False Positive = guest relocation cost, False Negative = lost room revenue) compared to a scenario without a model, and give business recommendations based on cancellation probability

## 2. Data Sources
- `data_hotel_booking_demand.csv` – Main dataset: 83,573 reservations from a hotel in Portugal with 10 features and the target `is_canceled`
- [Hotel booking demand](https://www.kaggle.com/datasets/jessemostipak/hotel-booking-demand) (`hotel_bookings.csv`) – Same source as the main dataset with extra columns. Used to calculate average daily rate (ADR) and length of stay as the basis for error cost, and to measure how much missing features (e.g. *lead time*) limit the model
- [Hotel Reservations Dataset](https://www.kaggle.com/datasets/ahsan81/hotel-reservations-classification-dataset) (`Hotel Reservations.csv`) – 36,275 reservations from another hotel. Used to validate whether cancellation patterns found in EDA also apply elsewhere
- Country Mapping – ISO, Continent, Region (`continents2.csv`, Kaggle) – Mapping of ISO country codes to continent and sub-region. Used for feature engineering to summarize `country` (162 countries) into `sub_region`

## 3. Technologies Used
- Programming Language: Python (Pandas, NumPy)
- Visualization: Matplotlib, Seaborn
- Machine Learning: Scikit-learn (Pipeline, ColumnTransformer, RandomizedSearchCV, Random Forest, and other classifiers), SHAP
- Version Control: Git
- Others: Jupyter Notebook / Google Colab, Pickle

## 4. Project Structure

```
├── README.md          <- The top-level README for developers using this project.
├── data
│   ├── external       <- Data from third party sources.
│   ├── interim        <- Intermediate data that has been transformed.
│   ├── processed      <- The final, canonical data sets for modeling.
│   └── raw            <- The original, immutable data dump.
│
├── models             <- Trained and serialized models, model predictions, or model summaries
│
├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
│                         the creator's initials, and a short `-` delimited description, e.g.
│                         `1.0-jqp-initial-data-exploration`.
│
├── references         <- Data dictionaries, manuals, and all other explanatory materials.
│
├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
│   └── figures        <- Generated graphics and figures to be used in reporting
│
├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
│                         generated with `pip freeze > requirements.txt`
│
└── src                <- Source code for use in this project.

```

## 5. Summary of Finding
### 5.1 Business Insight
- **Final model:** Random Forest (tuned, `class_weight="balanced"`, threshold 0.37) with 10 booking features. On test data (16,715 bookings) it reaches **F2-score 0.814**, **recall 88.9%**, **precision 60.7%**, **ROC AUC 0.888**, and **PR AUC 0.843**. All Project Charter targets are met.
- **Business impact:** Total cost drops from ≈ **EUR 2.20 million** (no model) to ≈ **EUR 0.61 million** (with model), a saving of ≈ **EUR 1.59 million (±72%)**, or ±EUR 95 per booking. This is an upper bound because it assumes every room from a detected cancellation is resold. It is also ±44% cheaper than the extreme strategy of treating all bookings as canceled. Cost assumptions come from the supporting dataset (FP ≈ EUR 104, FN ≈ EUR 358).
- **Deposit type:** `Non Refund` bookings are canceled ±99%, far above `No Deposit` (±28%). The hotel keeps the deposit, but the room stays empty.
- **Market segment:** Groups (±61%) and Online TA (±37%) cancel most often. Direct and Corporate (±15–18%) are far more stable.
- **Guest commitment:** Bookings with no special request cancel ±48%, dropping to ±22% with 1 request. Bookings that request parking are never canceled in this data. This pattern also holds in another hotel's data.
- **Guest origin:** Domestic guests (Portugal) cancel ±56%, versus ±16–20% for GBR, FRA, and DEU. At sub-region level, Southern Europe cancels ±50%, while Western and Northern Europe cancel ±19–21%.
- **Cancellation history:** Guests with previous cancellations tend to cancel again (±92–94%). This pattern is **specific to this hotel** and not seen in the other hotel's data.
- **Most important features:** `market_segment` and `deposit_type` (ROC AUC drops ±0.10–0.12 when shuffled), followed by `total_of_special_requests`, `is_domestic`, `customer_type`, and `required_car_parking_spaces`.
- **Probability reliability:** Bookings with probability > 0.8 are truly canceled ±96%, while probability ≤ 0.2 cancels only ±3%. The 0.4–0.6 zone is uncertain (±36% actual cancellation).
- **Where the model is less reliable:** Direct, Corporate, and Complementary segments (low recall), Western Europe guests (recall 55%), and Online TA (precision ±49%, many false alarms).
- **Data limitation:** The main dataset has no *lead time*, arrival date, price, or length of stay. An experiment on `hotel_bookings.csv` shows adding these features raises ROC AUC from ±0.89 to ±0.94, so the performance ceiling comes from the data, not the algorithm.

### 5.2 Actionable Recommendation
1. **Use tiered actions based on cancellation probability:**
   - **> 0.8:** room can enter the **overbooking / resale quota**
   - **0.37 – 0.8:** **reconfirm** with the guest (email/WhatsApp at D-7) and offer extra amenities. Do not overbook directly
   - **< 0.37:** treat normally
2. **Overbook per date using the sum of probabilities** of all bookings on that date as the estimated number of cancellations, keeping the overbooking number slightly below it as a safety margin.
3. **Review partnerships with agents/groups using Non Refund deposits.** Consider a *release date* (rooms released automatically if unconfirmed) or limit the room-block quota.
4. **Reduce risk from Online TA, domestic, and Southern Europe guests** with partial deposits, confirmation reminders, or incentives for **direct booking** (Direct cancels only ±14%).
5. **Encourage guest engagement before arrival** (room choice, parking, or special requests via pre-arrival email). This is a correlation, so the effect should be tested with an A/B test.
6. **Watch bookings made far in advance.** Supporting data shows bookings with lead time > 6 months cancel ±55–95%. Until this feature enters the model, add a reconfirmation rule for long-term bookings.
7. **Run a phased A/B test** before full rollout, e.g. use the model on 50% of dates and compare occupancy and relocation cost against the other 50%.
8. **Model development:** add `lead_time`, arrival month, `adr`, length of stay, and number of guests from the hotel's PMS (biggest impact); confirm when `required_car_parking_spaces` is recorded to rule out data leakage; replace the cost assumptions (`cost_fp`, `cost_fn`) with the hotel's real financial data and recompute the threshold; try other boosting algorithms (XGBoost, LightGBM, CatBoost) with probability calibration; monitor recall and precision monthly and retrain quarterly.

## 6. Contact
- Name: 
- Email:
- Linkedin:
