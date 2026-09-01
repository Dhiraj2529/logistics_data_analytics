# Data Dictionary — Olist Brazilian E-Commerce

This dataset contains real commercial data from 100k orders made between 2016 and 2018 across multiple marketplaces in Brazil.

------------------------------------------------

### 1. `olist_orders_dataset.csv`
<!-- Core table connecting customer orders, fulfillment statuses, and milestone timestamps. -->

| Column | Description | Data Type |
| --- | --- | --- |
| `order_id` | Unique identifier for each transaction | string |
| `customer_id` | Key to identify the customer context for this order | string |
| `order_status` | Status (`delivered`, `shipped`, `canceled`, `invoiced`, etc.) | string |
| `order_purchase_timestamp` | Timestamp when the purchase was made | datetime |
| `order_approved_at` | Timestamp when payment was approved | datetime |
| `order_delivered_carrier_date` | Timestamp when order was handed to logistics carrier | datetime |
| `order_delivered_customer_date` | Actual delivery timestamp to the customer | datetime |
| `order_estimated_delivery_date` | Estimated delivery date shown to customer at purchase | datetime |

------------------------------------------------

### 2. `olist_order_items_dataset.csv`
<!-- Item-level details linking products, sellers, prices, and shipping logistics. -->

| Column | Description | Data Type |
| --- | --- | --- |
| `order_id` | Identifier of the order | string |
| `order_item_id` | Sequential number identifying items in the same order | int |
| `product_id` | Unique product identifier | string |
| `seller_id` | Unique seller identifier | string |
| `shipping_limit_date` | Seller shipping limit date for carrier handover | datetime |
| `price` | Item selling price | float |
| `freight_value` | Item freight/shipping cost | float |

------------------------------------------------

### 3. `olist_order_payments_dataset.csv`
<!-- Payment installment and transaction options used per order. -->

| Column | Description | Data Type |
| --- | --- | --- |
| `order_id` | Identifier of the order | string |
| `payment_sequential` | Sequence index of the payment method | int |
| `payment_type` | Method (`credit_card`, `boleto`, `voucher`, `debit_card`) | string |
| `payment_installments` | Number of installments chosen | int |
| `payment_value` | Transaction amount | float |

------------------------------------------------

### 4. `olist_order_reviews_dataset.csv`
<!-- Customer review scores and feedback submitted after purchase. -->

| Column | Description | Data Type |
| --- | --- | --- |
| `review_id` | Unique identifier for the review | string |
| `order_id` | Identifier of the associated order | string |
| `review_score` | Rating from 1 to 5 | int |
| `review_comment_title` | Title of the review comment | string |
| `review_comment_message` | Written customer review text | string |
| `review_creation_date` | Timestamp when review survey was sent to customer | datetime |
| `review_answer_timestamp` | Timestamp when customer answered the review survey | datetime |

------------------------------------------------

### 5. `olist_products_dataset.csv`
<!-- Metadata, dimensions, and categories for purchased items. -->

| Column | Description | Data Type |
| --- | --- | --- |
| `product_id` | Unique product identifier | string |
| `product_category_name` | Root category name in Portuguese | string |
| `product_name_lenght` | Character count of product name | float |
| `product_description_lenght` | Character count of product description | float |
| `product_photos_qty` | Number of product photos published | float |
| `product_weight_g` | Product weight in grams | float |
| `product_length_cm` | Product package length in centimeters | float |
| `product_height_cm` | Product package height in centimeters | float |
| `product_width_cm` | Product package width in centimeters | float |

------------------------------------------------

### 6. `olist_customers_dataset.csv`
<!-- Customer location and identifier mappings. -->

| Column | Description | Data Type |
| --- | --- | --- |
| `customer_id` | Key to identify the order's specific customer instance | string |
| `customer_unique_id` | Unique identifier of the actual customer entity | string |
| `customer_zip_code_prefix` | First 5 digits of customer zip code | string |
| `customer_city` | Customer city name | string |
| `customer_state` | Customer state code (e.g., SP, RJ, MG) | string |

------------------------------------------------

### 7. `olist_sellers_dataset.csv`
<!-- Seller location and identifier information. -->

| Column | Description | Data Type |
| --- | --- | --- |
| `seller_id` | Unique seller identifier | string |
| `seller_zip_code_prefix` | First 5 digits of seller zip code | string |
| `seller_city` | Seller city name | string |
| `seller_state` | Seller state code (e.g., SP, PR, SC) | string |

------------------------------------------------

### 8. `olist_geolocation_dataset.csv`
<!-- Geographic coordinates and boundaries for Brazilian zip codes. -->

| Column | Description | Data Type |
| --- | --- | --- |
| `geolocation_zip_code_prefix` | First 5 digits of zip code | string |
| `geolocation_lat` | Latitude coordinate | float |
| `geolocation_lng` | Longitude coordinate | float |
| `geolocation_city` | City name | string |
| `geolocation_state` | State abbreviation | string |

------------------------------------------------

### 9. `product_category_name_translation.csv`
*Translation mapping table from Portuguese to English product categories.*

| Column | Description | Data Type |
| --- | --- | --- |
| `product_category_name` | Category name in Portuguese | string |
| `product_category_name_english` | Translated category name in English | string |