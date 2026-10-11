#!/data/data/com.termux/files/usr/bin/bash
# Run while app.py is running (use a second Termux session).
URL=http://127.0.0.1:5000/predict

echo "1) Valid request:"
curl -s -X POST $URL -H "Content-Type: application/json" \
  -d '{"study_hours": 6, "attendance_percentage": 85}'

echo; echo "2) Invalid attendance (150):"
curl -s -X POST $URL -H "Content-Type: application/json" \
  -d '{"study_hours": 6, "attendance_percentage": 150}'

echo; echo "3) Missing attendance:"
curl -s -X POST $URL -H "Content-Type: application/json" \
  -d '{"study_hours": 6}'
echo
