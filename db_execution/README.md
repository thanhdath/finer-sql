# SQLite API with Dataset Support

This API allows executing SQL queries on Spider and Bird datasets in a sandbox mode with transaction rollback.

## Features

- **Dataset Support**: Works with both Spider and Bird datasets
- **Sandbox Mode**: All operations run in transactions that are automatically rolled back
- **Pandas Integration**: Returns results as pandas DataFrame strings
- **Safe Execution**: Prevents database modifications through transaction rollback

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Start the API server:
```bash
cd db_execution
./run_api.sh
```

The API will be available at `http://localhost:8000`

## Usage

### API Endpoint: POST /execute

**Request Body:**
```json
{
    "dataset_name": "spider" | "bird",
    "db_id": "database_folder_name",
    "sql": "SELECT * FROM table_name LIMIT 5",
    "mode": "sandbox_rollback",
    "timeout_ms": 5000,
    "max_rows": 100
}
```

**Response:**
```json
{
    "ok": true,
    "statement_type": "SELECT",
    "rows": [...],
    "row_count": 5,
    "pandas_result": "formatted_table_string",
    "notice": "Executed on sandbox copy; changes rolled back."
}
```

### Test Commands

1. **Simple test command:**
```bash
cd db_execution
python test_command.py spider academic "SELECT * FROM student LIMIT 5"
python test_command.py bird address "SELECT * FROM Address LIMIT 5"
```

2. **Comprehensive test:**
```bash
cd db_execution
python test_api.py
```

## Dataset Structure

- **Spider datasets**: `/home/datht/mats/data/spider/database/{db_id}/{db_id}.sqlite`
- **Bird datasets**: `/home/datht/mats/data/bird/train/train_databases/{db_id}/{db_id}.sqlite`

## Examples

### Spider Dataset Example
```bash
curl -X POST "http://localhost:8000/execute" \
  -H "Content-Type: application/json" \
  -d '{
    "dataset_name": "spider",
    "db_id": "academic",
    "sql": "SELECT * FROM student LIMIT 5",
    "mode": "sandbox_rollback"
  }'
```

### Bird Dataset Example
```bash
curl -X POST "http://localhost:8000/execute" \
  -H "Content-Type: application/json" \
  -d '{
    "dataset_name": "bird",
    "db_id": "address",
    "sql": "SELECT * FROM Address LIMIT 5",
    "mode": "sandbox_rollback"
  }'
```

## Safety Features

- **Transaction Rollback**: All changes are automatically rolled back
- **Path Validation**: Prevents directory traversal attacks
- **Read-Only Mode**: Optional read-only mode for SELECT queries
- **Timeout Protection**: Configurable query timeouts
- **Row Limits**: Prevents excessive result sets

## Available Datasets

### Spider Datasets
- academic, activity_1, aircraft, allergy_1, apartment_rentals, architecture, assets_maintenance, baseball_1, battle_death, behavior_monitoring, bike_1, body_builder, book_2, browser_web, candidate_poll, car_1, chinook_1, cinema, city_record, climbing, club_1, coffee_shop, college_1, college_2, college_3, company_1, company_employee, company_office, concert_singer, county_public_safety, course_teach, cre_Doc_Control_Systems, cre_Doc_Template_Mgt, cre_Doc_Tracking_DB, cre_Docs_and_Epenses, cre_Drama_Workshop_Groups, cre_Theme_park, csu_1, culture_company, customer_complaints, customer_deliveries, customers_and_addresses, customers_and_invoices, customers_and_products_contacts, customers_campaigns_ecommerce, customers_card_transactions, debate, decoration_competition, department_management, department_store, device, document_management, dog_kennels, dorm_1, driving_school, e_government, e_learning, election, election_representative, employee_hire_evaluation, entertainment_awards, entrepreneur, epinions_1, farm, film_rank, flight_1, flight_2, flight_4, flight_company, formula_1, game_1, game_injury, gas_company, geo, gymnast, hospital_1, hr_1, icfp_1, imdb, inn_1, insurance_and_eClaims, insurance_fnol, insurance_policies, journal_committee, loan_1, local_govt_and_lot, local_govt_in_alabama, local_govt_mdm, machine_repair, manufactory_1, manufacturer, match_season, medicine_enzyme_interaction, mountain_photos, movie_1, museum_visit, music_1, music_2, music_4, musical, network_1, network_2, new_concert_singer, new_orchestra, new_pets_1, news_report, orchestra, party_host, party_people, performance_attendance, perpetrator, pets_1, phone_1, phone_market, pilot_record, poker_player, product_catalog, products_for_hire, products_gen_characteristics, program_share, protein_institute, race_track, railway, real_estate_properties, restaurant_1, restaurants, riding_club, roller_coaster, sakila_1, scholar, school_bus, school_finance, school_player, scientist_1, ship_1, ship_mission, shop_membership, singer, small_bank_1, soccer_1, soccer_2, solvency_ii, sports_competition, station_weather, store_1, store_product, storm_record, student_1, student_assessment, student_transcripts_tracking, swimming, theme_gallery, tracking_grants_for_research, tracking_orders, tracking_share_transactions, tracking_software_problems, train_station, tvshow, twitter_1, university_basketball, voter_1, voter_2, wedding, wine_1, workshop_paper, world_1, wrestler, wta_1, yelp

### Bird Datasets
- address, airline, app_store, authors, beer_factory, bike_share_1, book_publishing_company, books, car_retails, cars, chicago_crime, citeseer, codebase_comments, coinmarketcap, college_completion, computer_student, cookbook, craftbeer, cs_semester, disney, donor, european_football_1, food_inspection, food_inspection_2, genes, hockey, human_resources, ice_hockey_draft, image_and_language, language_corpus, law_episode, legislator, mental_health_survey, menu, mondial_geo, movie, movie_3, movie_platform, movielens, movies_4, music_platform_2, music_tracker, olympics, professional_basketball, public_review_platform, regional_sales, restaurant, retail_complains, retail_world, retails, sales, sales_in_weather, shakespeare, shipping, shooting, simpson_episodes, soccer_2016, social_media, software_company, student_loan, superstore, synthea, talkingdata, trains, university, video_games, works_cycles, world, world_development_indicators