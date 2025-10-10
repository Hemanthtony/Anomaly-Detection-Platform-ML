# TODO: Fix Weather and Spam Anomaly Detection Issues

## Issues Identified
1. Port mismatch: Frontend calls port 8002, backend runs on 8003
2. Predict functions return summary format, but app.py expects detailed anomaly lists
3. Spam model training fails with string labels ("spam", "ham") instead of numeric
4. Data alignment may fail if columns don't match model features

## Steps to Fix
- [ ] Update weather_anomaly.html to call port 8003
- [ ] Update spam_detection.html to call port 8003
- [ ] Fix spam_detection.py to handle string labels in training
- [ ] Update predict functions in weather_anomaly.py, spam_detection.py, network_anomaly.py to return detailed anomaly results
- [ ] Improve error handling in app.py for data alignment
- [ ] Test the fixes by running backend and testing frontend

## Progress
- [x] Identified issues
- [x] Created plan
- [x] Updated frontend ports
- [x] Fixed spam model training
- [x] Updated predict functions
- [x] Improved error handling
- [x] Tested fixes
