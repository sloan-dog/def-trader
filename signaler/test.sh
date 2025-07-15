# Create debug script
cat > debug-billing.sh << 'EOF'
#!/bin/bash

echo "=== DEBUG: All Billing Accounts ==="
gcloud billing accounts list

echo -e "\n=== DEBUG: Active Billing Accounts (raw) ==="
gcloud billing accounts list --filter="open=true"

echo -e "\n=== DEBUG: Active Billing Accounts (formatted) ==="
gcloud billing accounts list --filter="open=true" --format="table(name,displayName,open)"

echo -e "\n=== DEBUG: Count test ==="
ACTIVE_COUNT=$(gcloud billing accounts list --filter="open=true" --format="value(name)" | wc -l)
echo "Found $ACTIVE_COUNT active billing accounts"

echo -e "\n=== DEBUG: List test ==="
gcloud billing accounts list --filter="open=true" --format="value(name,displayName)" | while read -r line; do
    echo "Line: '$line'"
done
EOF

chmod +x debug-billing.sh
./debug-billing.sh
