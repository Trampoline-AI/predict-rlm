.PHONY: dataset clean

DATA_DIR := data
VERIFIED_DIR := $(DATA_DIR)/spreadsheetbench_verified_400
FULL_DIR := $(DATA_DIR)/all_data_912_v0.1
TRAINSET_DIR := $(DATA_DIR)/trainset
TESTSET_LINK := $(DATA_DIR)/testset

dataset:
	uv run python -m spreadsheet_rlm.bench.prepare_dataset

clean:
	rm -rf $(VERIFIED_DIR) $(FULL_DIR) $(TRAINSET_DIR)
	rm -f $(TESTSET_LINK)
