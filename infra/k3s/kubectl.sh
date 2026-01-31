#!/usr/bin/env sh
set -e
exec kubectl --kubeconfig "$(dirname "$0")/kubeconfig/kubeconfig.yaml" "$@"
