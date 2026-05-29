BINARY=terraform-provider-distllm
PLUGIN_DIR=~/.terraform.d/plugins/registry.terraform.io/distributed-llm/distllm/0.1.0/linux_amd64

.PHONY: build install clean

build:
	go build -o $(BINARY)

install: build
	mkdir -p $(PLUGIN_DIR)
	cp $(BINARY) $(PLUGIN_DIR)/$(BINARY)
	@echo "Provider installed. Add to .terraformrc:"
	@echo 'provider_installation {'
	@echo '  filesystem_mirror {'
	@echo '    path = "~/.terraform.d/plugins"'
	@echo '  }'
	@echo '}'

clean:
	rm -f $(BINARY)
