-- Daily Conversion Rate & Traffic Split by Group
SELECT 
    timestamp::DATE AS test_date,
    group_name,
    COUNT(DISTINCT user_id) AS total_users,
    SUM(converted) AS conversions,
    ROUND(AVG(converted) * 100.0, 2) AS conversion_rate_pct,
    ROUND(AVG(CASE WHEN converted = 1 THEN order_value END), 2) AS avg_order_value
FROM ab_test_checkout_data
GROUP BY 1, 2
ORDER BY 1 ASC, 2;
