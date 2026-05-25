-- =========================================
-- TARGET BRAZIL SQL ANALYSIS
-- CUSTOMER DISTRIBUTION ANALYSIS
-- =========================================


-- 1. Customer Distribution Across States

SELECT
    customer_state,
    COUNT(customer_id) AS total_customers
FROM customers
GROUP BY customer_state
ORDER BY total_customers DESC;


-- 2. Month-on-Month Orders by State

SELECT
    customers.customer_state,
    
    YEAR(orders.order_purchase_timestamp) AS order_year,
    
    MONTH(orders.order_purchase_timestamp) AS order_month,
    
    COUNT(orders.order_id) AS total_orders

FROM orders

JOIN customers
    ON orders.customer_id = customers.customer_id

GROUP BY
    customers.customer_state,
    order_year,
    order_month

ORDER BY
    customers.customer_state,
    order_year,
    order_month;


-- 3. Top 10 States by Total Orders

SELECT
    customers.customer_state,
    COUNT(orders.order_id) AS total_orders

FROM orders

JOIN customers
    ON orders.customer_id = customers.customer_id

GROUP BY customers.customer_state

ORDER BY total_orders DESC

LIMIT 10;
