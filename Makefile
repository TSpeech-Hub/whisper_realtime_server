PROTO_DIR=proto
GEN_DIR=src/generated

.PHONY: proto clean

proto:
	mkdir -p $(GEN_DIR)
	touch $(GEN_DIR)/__init__.py
	python3 -m grpc_tools.protoc \
		-I$(PROTO_DIR) \
		--python_out=$(GEN_DIR) \
		--grpc_python_out=$(GEN_DIR) \
		$(PROTO_DIR)/*.proto

	python3 fix_proto_imports.py

clean:
	rm -rf $(GEN_DIR)/*.py
