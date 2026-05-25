-- =========================================
-- TARGET BRAZIL SQL ANALYSIS
-- ORDERS TREND ANALYSIS
-- =========================================


-- 1. Yearly Growth in Orders

SELECT 
    YEAR(order_purchase_timestamp) AS order_year,
    COUNT(order_id) AS total_orders
FROM orders
GROUP BY order_year
ORDER BY order_year;


-- 2. Monthly Seasonality Analysis

SELECT 
    YEAR(order_purchase_timestamp) AS order_year,
    MONTH(order_purchase_timestamp) AS order_month,
    COUNT(order_id) AS total_orders
FROM orders
GROUP BY order_year, order_month
ORDER BY order_year, order_month;


-- 3. Order Distribution by Time of Day

SELECT
    CASE
        WHEN HOUR(order_purchase_timestamp) BETWEEN 0 AND 6 THEN 'Dawn'
        WHEN HOUR(order_purchase_timestamp) BETWEEN 7 AND 12 THEN 'Morning'
        WHEN HOUR(order_purchase_timestamp) BETWEEN 13 AND 18 THEN 'Afternoon'
        ELSE 'Night'
    END AS time_of_day,
    
    COUNT(order_id) AS total_orders

FROM orders

GROUP BY time_of_day

ORDER BY total_orders DESC;
