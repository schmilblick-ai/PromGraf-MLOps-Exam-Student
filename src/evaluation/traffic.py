from run_evaluation import generate_traffic, _process_data, _fetch_data
import random
# --- Main execution logic ---
if __name__ == "__main__":
    _full_data_cache = _process_data(_fetch_data())
    alea=random.randint(1,60)
    generate_traffic(alea,_full_data_cache)
