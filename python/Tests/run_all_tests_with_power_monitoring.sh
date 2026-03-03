#!/bin/bash

###############################################################################
# Test Runner with Power Monitoring
# 
# This script runs all three test configurations (GPU, AVX2, CPU) sequentially
# and monitors system power consumption using Intel RAPL during each run.
#
# Requirements: Python 3, openEMS
# Power Monitoring: Intel RAPL via /sys/class/powercap/
###############################################################################

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORK_DIR="$(pwd)"
TEMP_DIR=$(mktemp -d)
SUMMARY_FILE="$WORK_DIR/power_comparison_summary.txt"

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Test definitions
declare -a TESTS=(
    "Rect_Waveguide_W_Local_Absorbers.py"
    "Rect_Waveguide_W_Local_Absorbers_avx2.py"
    "Rect_Waveguide_W_Local_Absorbers_cpu.py"
)

declare -a TEST_NAMES=(
    "GPU Engine"
    "AVX2 Multithreaded"
    "SSE Multithreaded (CPU)"
)

# Power monitoring functions
check_rapl_available() {
    if [ ! -d "/sys/class/powercap/intel-rapl" ]; then
        echo -e "${RED}[ERROR] Intel RAPL not available on this system${NC}"
        return 1
    fi
    return 0
}

read_rapl_energy() {
    local domain=$1
    local energy_file="/sys/devices/virtual/powercap/$domain/energy_uj"
    
    if [ ! -r "$energy_file" ]; then
        echo "0"
        return 1
    fi
    
    cat "$energy_file" 2>/dev/null || echo "0"
}

get_rapl_domains() {
    local package_domain="intel-rapl:0"
    local core_domain="intel-rapl:0:0"
    local uncore_domain="intel-rapl:0:1"
    
    # Check which domains are available and readable
    local domains=""
    if [ -r "/sys/devices/virtual/powercap/$package_domain/energy_uj" ]; then
        domains="$package_domain"
    fi
    if [ -r "/sys/devices/virtual/powercap/$core_domain/energy_uj" ]; then
        domains="$domains $core_domain"
    fi
    if [ -r "/sys/devices/virtual/powercap/$uncore_domain/energy_uj" ]; then
        domains="$domains $uncore_domain"
    fi
    
    echo "$domains"
}

monitor_power() {
    local output_file=$1
    local test_pid=$2
    local interval=${3:-0.5}

    > "$output_file"

    local domains
    domains=$(get_rapl_domains)

    # Write header row
    {
        echo -n "# elapsed_s"
        for domain in $domains; do
            echo -n " ${domain}_W"
        done
        echo ""
    } >> "$output_file"

    # Read initial energy values and timestamp
    declare -A prev_energy
    for domain in $domains; do
        prev_energy[$domain]=$(read_rapl_energy "$domain")
    done
    local prev_ns
    prev_ns=$(date +%s%N)
    local start_ns=$prev_ns

    sleep "$interval"

    # Monitor while process is running
    while kill -0 "$test_pid" 2>/dev/null; do
        local now_ns
        now_ns=$(date +%s%N)
        local dt_ns=$(( now_ns - prev_ns ))
        local elapsed_s
        elapsed_s=$(echo "scale=3; ($now_ns - $start_ns) / 1000000000" | bc)

        {
            echo -n "$elapsed_s"
            for domain in $domains; do
                local cur_uj
                cur_uj=$(read_rapl_energy "$domain")
                local prev_uj=${prev_energy[$domain]}
                local delta_uj
                if [ "$cur_uj" -ge "$prev_uj" ]; then
                    delta_uj=$(( cur_uj - prev_uj ))
                else
                    # RAPL counter wraparound - skip sample
                    delta_uj=0
                fi
                # Power (W) = delta_microjoules / delta_nanoseconds * 1000
                # because: µJ/ns = W * 1e-6 / 1e-9 = W * 1e3
                local power_w
                power_w=$(echo "scale=2; $delta_uj * 1000 / $dt_ns" | bc)
                echo -n " $power_w"
                prev_energy[$domain]=$cur_uj
            done
            echo ""
        } >> "$output_file"

        prev_ns=$now_ns
        sleep "$interval"
    done
}

calculate_power_stats() {
    local power_log=$1
    local results_dir=$2

    if [ ! -f "$power_log" ]; then
        return
    fi

    # Column 2 is the package-0 domain (total CPU+uncore power).
    # Remaining columns are sub-domains (core, uncore) which are already
    # included in the package total, so we only use column 2 to avoid
    # double-counting.
    awk -v dir="$results_dir" '
    /^#/ { next }  # skip header/comment lines
    NF >= 2 {
        power = $2 + 0
        ts    = $1 + 0

        if (count == 0) {
            min_power = power
            max_power = power
            first_ts  = ts
        }

        if (power > max_power) max_power = power
        if (power < min_power) min_power = power
        sum_power += power
        last_ts = ts
        count++
    }
    END {
        if (count > 0) {
            avg_power    = sum_power / count
            duration     = last_ts - first_ts
            total_energy = avg_power * duration
            printf "duration_s=%.1f\n",      duration      > (dir "/power_stats")
            printf "avg_power_W=%.1f\n",     avg_power    >> (dir "/power_stats")
            printf "max_power_W=%.1f\n",     max_power    >> (dir "/power_stats")
            printf "min_power_W=%.1f\n",     min_power    >> (dir "/power_stats")
            printf "total_energy_J=%.1f\n",  total_energy >> (dir "/power_stats")
            printf "samples=%d\n",           count        >> (dir "/power_stats")
        }
    }
    ' "$power_log"
}

run_test() {
    local test_script=$1
    local test_name=$2
    local test_index=$3
    
    echo -e "${BLUE}\n========================================${NC}"
    echo -e "${BLUE}Running Test $test_index: $test_name${NC}"
    echo -e "${BLUE}========================================${NC}\n"
    
    # Check if script exists
    if [ ! -f "$SCRIPT_DIR/$test_script" ]; then
        echo -e "${RED}[ERROR] Test script not found: $test_script${NC}"
        return 1
    fi
    
    # Create results directory early so we can save logs there
    local results_dir="$WORK_DIR/${test_script%.*}"
    if [ ! -d "$results_dir" ]; then
        mkdir -p "$results_dir"
    fi
    
    local log_file="$results_dir/${test_script%.*}_output.log"
    local power_log="$results_dir/${test_script%.*}_power.log"
    local start_time=$(date +%s.%N)
    
    echo -e "${YELLOW}[INFO] Starting test at $(date)${NC}"
    echo -e "${YELLOW}[INFO] Command: python3 $test_script${NC}"
    
    # Run the test in background to monitor power simultaneously
    (
        cd "$SCRIPT_DIR"
        python3 "$test_script" 2>&1
    ) > "$log_file" 2>&1 &
    
    local test_pid=$!
    echo -e "${YELLOW}[INFO] Test process ID: $test_pid${NC}"
    
    # Start power monitoring in background
    monitor_power "$power_log" $test_pid 0.5 &
    local monitor_pid=$!
    
    # Wait for test to complete
    if wait $test_pid; then
        echo -e "${GREEN}[OK] Test completed successfully${NC}"
    else
        echo -e "${RED}[ERROR] Test failed with exit code $?${NC}"
    fi
    
    # Wait for monitoring to finish
    wait $monitor_pid 2>/dev/null || true
    
    local end_time=$(date +%s.%N)
    local elapsed=$(echo "$end_time - $start_time" | bc)
    
    echo -e "${YELLOW}[INFO] Elapsed time: ${elapsed}s${NC}"
    echo -e "${YELLOW}[INFO] Test log saved to: $log_file${NC}"
    echo -e "${YELLOW}[INFO] Power log saved to: $power_log${NC}"
    
    # Move Python test results from /tmp to local results directory
    local tmp_results_dir="/tmp/${test_script%.*}"
    if [ -d "$tmp_results_dir" ]; then
        # Copy PNG files to results directory
        cp "$tmp_results_dir"/*.png "$results_dir"/ 2>/dev/null || true
        # Copy power stats if they exist
        if [ -f "$tmp_results_dir/power_stats" ]; then
            cp "$tmp_results_dir/power_stats" "$results_dir/"
        fi
    fi
    
    # Calculate power statistics
    calculate_power_stats "$power_log" "$results_dir"
    
    echo -e "${YELLOW}[INFO] Results saved to: $results_dir${NC}"
    
    # Store test result info (properly quoted to avoid issues when sourcing)
    echo "elapsed_time='$elapsed'" >> "$TEMP_DIR/${test_script%.*}_info"
    echo "test_name='$test_name'" >> "$TEMP_DIR/${test_script%.*}_info"
    echo "results_dir='$results_dir'" >> "$TEMP_DIR/${test_script%.*}_info"
}

generate_summary() {
    echo -e "\n${BLUE}========================================${NC}"
    echo -e "${BLUE}Generating Summary Report${NC}"
    echo -e "${BLUE}========================================${NC}\n"
    
    {
        echo "=================================================================================="
        echo "OpenEMS Test Comparison Report - Power & Performance"
        echo "Generated: $(date)"
        echo "=================================================================================="
        echo ""
        
        for i in "${!TESTS[@]}"; do
            local test_script="${TESTS[$i]}"
            local test_name="${TEST_NAMES[$i]}"
            local info_file="$TEMP_DIR/${test_script%.*}_info"
            
            echo "Test $((i+1)): $test_name"
            echo "Script: $test_script"
            echo "---"
            
            if [ -f "$info_file" ]; then
                source "$info_file"
                local results_dir="${results_dir:-}"
                echo "Execution Time: ${elapsed_time}s"
                echo "Results Directory: $results_dir"
            else
                local results_dir="$WORK_DIR/${test_script%.*}"
            fi
            
            local power_stats="$results_dir/power_stats"
            
            if [ -f "$power_stats" ]; then
                echo ""
                echo "Power Consumption Statistics:"
                cat "$power_stats" | sed 's/^/  /'
            fi
            
            if [ -d "$results_dir" ]; then
                echo ""
                echo "Generated Files:"
                ls -lh "$results_dir"/*.png 2>/dev/null | awk '{print "  " $9 " (" $5 ")"}' || echo "  (no PNG files found)"
                
                echo ""
                echo "Log Files:"
                ls -lh "$results_dir"/*.log 2>/dev/null | awk '{print "  " $9 " (" $5 ")"}' || echo "  (no log files found)"
            fi
            
            echo ""
            echo "=================================================================================="
            echo ""
        done
        
    } | tee "$SUMMARY_FILE"
    
    echo -e "${GREEN}[OK] Summary report saved to: $SUMMARY_FILE${NC}"
}

print_help() {
    cat << EOF
Usage: $(basename "$0") [OPTIONS]

Run all OpenEMS test configurations with power monitoring.

Options:
    -h, --help          Show this help message
    -c, --check         Check for RAPL availability and exit
    -v, --verbose       Enable verbose output
    
Examples:
    $(basename "$0")              # Run all tests
    $(basename "$0") --check       # Check RAPL availability
    
EOF
}

# Main execution
main() {
    echo -e "${GREEN}========================================${NC}"
    echo -e "${GREEN}OpenEMS Test Suite with Power Monitoring${NC}"
    echo -e "${GREEN}========================================${NC}\n"
    
    # Check for RAPL availability
    if ! check_rapl_available; then
        echo -e "${YELLOW}[WARNING] Power monitoring via RAPL not available${NC}"
        echo -e "${YELLOW}[WARNING] Tests will run but power data will not be collected${NC}\n"
    else
        echo -e "${GREEN}[OK] Intel RAPL power monitoring available${NC}\n"
    fi
    
    # Check if Python is available
    if ! command -v python3 &> /dev/null; then
        echo -e "${RED}[ERROR] python3 not found in PATH${NC}"
        exit 1
    fi
    
    # Run all tests
    local test_count=${#TESTS[@]}
    echo -e "${BLUE}Running $test_count tests...${NC}\n"
    
    for i in "${!TESTS[@]}"; do
        run_test "${TESTS[$i]}" "${TEST_NAMES[$i]}" $((i+1)) || true
    done
    
    # Generate summary
    generate_summary
    
    # Cleanup
    rm -rf "$TEMP_DIR"
    
    echo -e "\n${GREEN}[OK] All tests completed!${NC}"
    echo -e "${GREEN}Check the following for detailed results:${NC}"
    echo -e "${GREEN}  - Individual result folders in current directory: $WORK_DIR${NC}"
    echo -e "${GREEN}  - Summary report: $SUMMARY_FILE${NC}\n"
}

# Parse command line arguments
case "${1:-}" in
    -h|--help)
        print_help
        exit 0
        ;;
    -c|--check)
        if check_rapl_available; then
            echo -e "${GREEN}[OK] Intel RAPL is available${NC}"
            echo -e "${GREEN}Available domains:${NC}"
            get_rapl_domains | tr ' ' '\n' | sed 's/^/  /'
            exit 0
        else
            exit 1
        fi
        ;;
    -v|--verbose)
        set -x
        main
        ;;
    "")
        main
        ;;
    *)
        echo -e "${RED}[ERROR] Unknown option: $1${NC}"
        print_help
        exit 1
        ;;
esac
