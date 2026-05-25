-- =========================================
-- TARGET BRAZIL SQL ANALYSIS
-- DATA EXPLORATION
-- =========================================


-- 1. View Structure of Customers Table

DESCRIBE customers;


-- 2. Find Order Purchase Time Range

SELECT 
    MIN(order_purchase_timestamp) AS first_order_date,
    MAX(order_purchase_timestamp) AS latest_order_date
FROM orders;


-- 3. Count Total Cities and States

SELECT 
    COUNT(DISTINCT customer_city) AS total_cities,
    COUNT(DISTINCT customer_state) AS total_states
FROM customers
JOIN orders
    ON customers.customer_id = orders.customer_id;
