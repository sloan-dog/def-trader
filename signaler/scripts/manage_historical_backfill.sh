#!/bin/bash
# Script to manage historical backfills

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Default values
START_YEAR=1995
END_YEAR=$(date +%Y)
BATCH_SIZE=20
SERVICE_URL=""

# Function to print colored output
print_error() { echo -e "${RED}ERROR: $1${NC}" >&2; }
print_success() { echo -e "${GREEN}SUCCESS: $1${NC}"; }
print_info() { echo -e "${YELLOW}INFO: $1${NC}"; }

# Function to show usage
show_usage() {
    cat << EOF
Usage: $0 [COMMAND] [OPTIONS]

Commands:
    start       Start a new historical backfill
    status      Check status of a backfill
    active      List all active backfills
    local       Run backfill locally (not via API)

Options:
    --start-year YEAR       Start year for backfill (default: 1995)
    --end-year YEAR         End year for backfill (default: current year)
    --batch-size SIZE       Batch size for processing (default: 20)
    --backfill-id ID        Backfill ID (for status/resume)
    --service-url URL       Backfill service URL
    --year-range SIZE       Process SIZE years at a time (default: 5)

Examples:
    # Start backfill for 1995-2000
    $0 start --start-year 1995 --end-year 2000 --service-url https://backfill-service.run.app

    # Check status
    $0 status --backfill-id historical_1995_2000_20250115_123456 --service-url https://backfill-service.run.app

    # Run locally
    $0 local --start-year 2020 --end-year 2024
EOF
}

# Parse command
COMMAND=$1
shift || true

# Parse options
while [[ $# -gt 0 ]]; do
    case $1 in
        --start-year)
            START_YEAR="$2"
            shift 2
            ;;
        --end-year)
            END_YEAR="$2"
            shift 2
            ;;
        --batch-size)
            BATCH_SIZE="$2"
            shift 2
            ;;
        --backfill-id)
            BACKFILL_ID="$2"
            shift 2
            ;;
        --service-url)
            SERVICE_URL="$2"
            shift 2
            ;;
        --year-range)
            YEAR_RANGE="$2"
            shift 2
            ;;
        --help)
            show_usage
            exit 0
            ;;
        *)
            print_error "Unknown option: $1"
            show_usage
            exit 1
            ;;
    esac
done

# Execute command
case $COMMAND in
    start)
        if [[ -z "$SERVICE_URL" ]]; then
            print_error "Service URL is required for start command"
            exit 1
        fi
        
        print_info "Starting historical backfill: $START_YEAR to $END_YEAR"
        
        # Split into year ranges if needed
        YEAR_RANGE=${YEAR_RANGE:-5}
        current_start=$START_YEAR
        
        while [ $current_start -le $END_YEAR ]; do
            current_end=$((current_start + YEAR_RANGE - 1))
            if [ $current_end -gt $END_YEAR ]; then
                current_end=$END_YEAR
            fi
            
            print_info "Triggering backfill for $current_start-$current_end"
            
            response=$(curl -s -X POST "$SERVICE_URL/backfill/historical" \
                -H "Content-Type: application/json" \
                -d "{
                    \"start_year\": $current_start,
                    \"end_year\": $current_end,
                    \"batch_size\": $BATCH_SIZE,
                    \"data_types\": [\"ohlcv\"]
                }")
            
            echo "$response" | jq '.' || echo "$response"
            
            current_start=$((current_end + 1))
            
            # Wait a bit between requests
            if [ $current_start -le $END_YEAR ]; then
                sleep 5
            fi
        done
        
        print_success "All backfill requests submitted"
        ;;
        
    status)
        if [[ -z "$SERVICE_URL" ]] || [[ -z "$BACKFILL_ID" ]]; then
            print_error "Service URL and backfill ID are required for status command"
            exit 1
        fi
        
        print_info "Checking status for backfill: $BACKFILL_ID"
        
        response=$(curl -s "$SERVICE_URL/backfill/status/$BACKFILL_ID")
        
        # Pretty print the response
        echo "$response" | jq '.' || echo "$response"
        
        # Extract key metrics
        if command -v jq &> /dev/null; then
            status=$(echo "$response" | jq -r '.status // "unknown"')
            progress=$(echo "$response" | jq -r '.progress_percentage // 0')
            completed=$(echo "$response" | jq -r '.completed_months // 0')
            total=$(echo "$response" | jq -r '.total_months // 0')
            
            print_info "Status: $status"
            print_info "Progress: ${progress}% ($completed/$total months)"
            
            if [ "$status" = "in_progress" ]; then
                eta=$(echo "$response" | jq -r '.estimated_hours_remaining // "unknown"')
                print_info "Estimated time remaining: $eta hours"
            fi
        fi
        ;;
        
    active)
        if [[ -z "$SERVICE_URL" ]]; then
            print_error "Service URL is required for active command"
            exit 1
        fi
        
        print_info "Fetching active backfills..."
        
        response=$(curl -s "$SERVICE_URL/backfill/active")
        
        # Pretty print the response
        echo "$response" | jq '.' || echo "$response"
        
        # Summary
        if command -v jq &> /dev/null; then
            count=$(echo "$response" | jq '.active_backfills | length')
            print_info "Total active backfills: $count"
        fi
        ;;
        
    local)
        print_info "Running backfill locally for $START_YEAR to $END_YEAR"
        
        # Run the Python script directly
        python -m src.jobs.historical_backfill_job \
            --start-year $START_YEAR \
            --end-year $END_YEAR \
            --batch-size $BATCH_SIZE \
            --data-types ohlcv
        ;;
        
    *)
        print_error "Unknown command: $COMMAND"
        show_usage
        exit 1
        ;;
esac