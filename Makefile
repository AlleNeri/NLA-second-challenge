APPTAINER = apptainer exec $$NLA_AMSC_CONTAINER
BUILD_DIR := build

.PHONY: run
run: $(BUILD_DIR)/main
	./$(BUILD_DIR)/main social.mtx

$(BUILD_DIR)/main: main.cpp
	mkdir -p $(BUILD_DIR)
	$(APPTAINER) bash -c 'g++ -I $${mkEigenInc} $< -o $@'

.PHONY: clean
clean:
	rm -rf $(BUILD_DIR)
	rm -f *.txt
	# remove also all the .mtx files except social.mtx
	find . -maxdepth 1 -type f -name '*.mtx' ! -name 'social.mtx' -delete
.PHONY: clear
clear: clean
