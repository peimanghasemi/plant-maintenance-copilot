# Maintenance policy for condition-based work orders

## Priority from remaining useful life (RUL)
- RUL below 48 hours: priority urgent, plan the intervention immediately and inform the shift leader.
- RUL between 48 and 120 hours: priority high, schedule within the next planned stop or within 3 days.
- RUL between 120 and 250 hours: priority medium, schedule within 2 weeks and order spare parts.
- RUL above 250 hours: no work order needed, keep monitoring.

## Approval
Condition-based work orders proposed by monitoring tools must be approved by the area maintenance engineer before they are created in the CMMS. Tools may propose, never create.

## Anomaly alarms
An anomaly alarm without a low RUL prediction is a reason to inspect, not to replace parts. Create an inspection work order with priority medium.
