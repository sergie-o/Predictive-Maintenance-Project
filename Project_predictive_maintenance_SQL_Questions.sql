--1: What is the distribution of machine failures?
SELECT  "machine failure", COUNT(*) AS total  FROM  cleaned_data GROUP BY  "machine failure";

--2. What are the average values of operational variables for successful vs. failed machines?
SELECT "machine failure",
       AVG("Air temperature [K]") AS avg_air_temp,
       AVG("Process temperature [K]") AS avg_process_temp,
       AVG("Torque [Nm]") AS avg_torque,
       AVG("Tool wear [min]") AS avg_tool_wear,
       AVG("Rotational speed [rpm]") AS avg_rot_speed
FROM cleaned_data
GROUP BY "machine failure";

--3. What is the failure rate across different product types?
SELECT "type",
       SUM("machine failure") * 100.0 / COUNT(*) AS failure_rate_percentage
FROM cleaned_data
GROUP BY "type";

-- 4. Are failures more common at high tool wear times?
SELECT "Tool wear [min]", "Machine failure" FROM cleaned_data ORDER BY "Tool wear [min]" DESC;

--5 What is the distribution of failure vs. success across time/tool wear?
SELECT "Tool wear [min]", COUNT(*) AS total, SUM("Machine failure") AS failures,
       COUNT(*) - SUM("Machine failure") AS success
FROM cleaned_data
GROUP BY "Tool wear [min]"
ORDER BY "Tool wear [min]";

--6 .Are there specific combinations of variables (e.g., low torque and high wear) that lead to failures?
SELECT * FROM cleaned_data WHERE "Torque [Nm]" < 25 AND "Tool wear [min]" > 200 AND "Machine failure" = 1;
 
 --7  Do certain failure types occur more frequently?
 SELECT
  SUM("TWF") AS TWF_count,
  SUM("HDF") AS HDF_count,
  SUM("PWF") AS PWF_count,
  SUM("OSF") AS OSF_count,
  SUM("RNF") AS RNF_count
FROM cleaned_data;

--8 . What are the operational conditions under which each heat dissipation failure HDF occurs?
SELECT * FROM cleaned_data WHERE "HDF" > 0;

--9  What are the operational conditions under which tool wear failures TWF occurs?
SELECT * FROM cleaned_data WHERE "TWF" > 0;

--10 What are the operational conditions for machine success?
SELECT * FROM cleaned_data WHERE "Machine Failure"> 0