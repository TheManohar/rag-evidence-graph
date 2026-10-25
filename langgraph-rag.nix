{ config, pkgs, lib, ... }:

# Bounded LangGraph RAG evidence graph -- declarative source of truth.
#
# This module is INERT unless explicitly imported AND enabled:
#
#   imports = [ ../modules/langgraph-rag.nix ];
#   services.langgraph-rag.enable = true;
#
# It declares its OWN interpreter with LangGraph. The existing hermes-rag
# services exec a different python (no langgraph); the shared system python is
# deliberately NOT widened here.
let
  cfg = config.services.langgraph-rag;

  # Dedicated interpreter: the SQLite checkpoint backend lives here.
  # The shared system-profile python is deliberately NOT widened.
  runtime = pkgs.python3.withPackages (ps: [
    ps.langgraph
    ps.langgraph-checkpoint-sqlite
  ]);

  sourceTree = pkgs.runCommand "langgraph-rag-tree" { } ''
    mkdir -p $out
    cp -r ${./src} $out/src
    chmod -R u+w $out
  '';

  common = {
    NoNewPrivileges = true;
    PrivateTmp = true;
    PrivateDevices = true;
    ProtectSystem = "strict";
    ProtectHome = true;
    ProtectKernelTunables = true;
    ProtectKernelModules = true;
    ProtectControlGroups = true;
    LockPersonality = true;
    RestrictAddressFamilies = [ "AF_INET" "AF_INET6" "AF_UNIX" ];
    RestrictNamespaces = true;
    SystemCallArchitectures = "native";
    User = "root";
    Group = "root";
  };
in
{
  options.services.langgraph-rag = {
    enable = lib.mkEnableOption
      "bounded LangGraph RAG evidence graph adapter (loopback 8791)";

    host = lib.mkOption {
      type = lib.types.str;
      default = "127.0.0.1";
      description = "Loopback address to bind. The adapter refuses non-loopback binds.";
    };

    port = lib.mkOption {
      type = lib.types.port;
      default = 8791;
      description = "Loopback port. Not added to the firewall; loopback only.";
    };

    indexVersion = lib.mkOption {
      type = lib.types.str;
      default = "rag-v1";
      description = "Expected index version; citations from any other version are dropped as stale.";
    };

    slaSeconds = lib.mkOption {
      type = lib.types.int;
      default = 900;
      description = "Freshness SLA in seconds (existing 15-minute SLA).";
    };

    knowledgeSearchUrl = lib.mkOption {
      type = lib.types.str;
      default = "http://127.0.0.1:8788/knowledge_search";
      description = "Existing read-only knowledge_search contract (unchanged).";
    };

    freshnessUrl = lib.mkOption {
      type = lib.types.str;
      default = "http://127.0.0.1:8790/freshness";
      description = ''
        Published freshness-contract read path (task t_de551615). The exact
        endpoint and field names must be reconciled with that published
        contract at admission.
      '';
    };

    checkpointPath = lib.mkOption {
      type = lib.types.str;
      default = "/var/lib/hermes-langgraph/checkpoints.sqlite3";
      description = ''
        File-backed SQLite checkpoint path: outside the Nix store, not on
        tmpfs, directory 0700, database (and -wal/-shm sidecars) 0600 via
        UMask=0077. Must be included in the /var/lib file-level backup;
        checkpoints are reconstructible service state and should be pruned
        periodically.
      '';
    };

    requestDeadline = lib.mkOption {
      type = lib.types.float;
      default = 5.0;
      description = ''
        Wall-clock budget for one /rag_answer request, in seconds. The adapter
        answers with `deadline_exceeded` (HTTP 504) rather than waiting longer,
        and each upstream call is shortened to the remaining budget. The
        default sits inside the 6.0 s rag-auto plugin hook deadline and replaces
        the historical ~21 s worst case (5 s retrieval x 3 attempts + 3 s
        freshness). 0 disables the deadline.
      '';
    };

    metricsPath = lib.mkOption {
      type = lib.types.str;
      default = "/var/lib/hermes-langgraph/rag-metrics.jsonl";
      description = ''
        Redacted JSONL provenance sink: exactly one line per /rag_answer
        request (ts, run_id, status, wall_ms, evidence_count, freshness_age,
        index_version, replayed, caller). No query text, snippets or auth
        material. Empty string disables the sink.
      '';
    };
  };

  config = lib.mkIf cfg.enable {
    systemd.services.langgraph-rag = {
      description = "Bounded LangGraph RAG evidence graph adapter (${cfg.host}:${toString cfg.port})";
      wantedBy = [ "multi-user.target" ];
      after = [ "network.target" "hermes-rag-search.service" "hermes-rag-projection.service" ];
      wants = [ "hermes-rag-search.service" ];
      serviceConfig = common // {
        Type = "simple";
        StateDirectory = "hermes-langgraph";
        StateDirectoryMode = "0700";
        UMask = "0077";
        WorkingDirectory = "${sourceTree}";
        Environment = [
          "PYTHONUNBUFFERED=1"
          "PYTHONDONTWRITEBYTECODE=1"
          "PYTHONPATH=${sourceTree}/src"
        ];
        ExecStart = "${runtime}/bin/python3 -m langgraph_rag.adapter"
          + " --host ${cfg.host} --port ${toString cfg.port}"
          + " --index-version ${cfg.indexVersion}"
          + " --sla-seconds ${toString cfg.slaSeconds}"
          + " --knowledge-search-url ${cfg.knowledgeSearchUrl}"
          + " --freshness-url ${cfg.freshnessUrl}"
          + " --checkpoint sqlite"
          + " --checkpoint-path ${cfg.checkpointPath}"
          + " --request-deadline ${builtins.toString cfg.requestDeadline}"
          + " --metrics-path ${cfg.metricsPath}";
        Restart = "on-failure";
        RestartSec = 3;
      };
    };
  };
}
