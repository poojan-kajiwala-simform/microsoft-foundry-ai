# Contoso Field Engineering Handbook: Incident Response

## Severity Levels

Severity 1 means a customer site is fully down or there is a safety risk. Severity 2 means a major function is degraded but a workaround exists. Severity 3 is a minor defect with no customer impact. When unsure, declare the higher severity and downgrade later.

## Response Targets

For a Severity 1 incident the on-call engineer must acknowledge within 15 minutes and post a status update every 60 minutes until resolved. Severity 2 incidents require acknowledgement within 2 hours and daily updates. Severity 3 incidents are handled in the next business day triage.

## On-Call Rotation

The on-call rotation changes every Monday at 09:00 local time and each shift lasts one week. Engineers may swap shifts by finding a replacement and recording the swap in the rota at least 48 hours in advance. On-call engineers receive a stipend of $400 per week of coverage.

## Escalation

If the on-call engineer has not resolved a Severity 1 incident within 2 hours, they must escalate to the engineering manager. If the manager is unreachable for 15 minutes, escalate to the regional director.

## Known Error Codes

Error code PX-4471 means the pump controller has lost its calibration. Power-cycle the controller, then run the recalibration routine from the service menu. Error code PX-9020 means the pressure sensor has failed and the unit must be replaced on site; do not attempt a field repair.

## Postmortems

A written postmortem is required within 5 business days of any Severity 1 incident. Postmortems are blameless and are stored in the engineering wiki.
