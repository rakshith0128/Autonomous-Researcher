import { useEffect, useMemo } from "react";
import {
  Background,
  Controls,
  type Edge,
  Handle,
  type Node,
  Position,
  ReactFlow,
  useEdgesState,
  useNodesState,
} from "@xyflow/react";
import "@xyflow/react/dist/style.css";
import { AGENTS, type AgentName } from "@/lib/types";
import type { RunState } from "@/hooks/useRunState";

/**
 * The agent graph, lit up as the run moves through it.
 *
 * This is the clearest evidence in the whole UI that the system is a state
 * machine rather than a prompt chain — you can watch the Critic send work
 * *backwards* along a red edge and the cycle counter tick. A linear progress
 * bar would show none of that.
 *
 * The topology is fetched from the backend rather than duplicated here, so the
 * picture cannot drift away from the graph that actually executes.
 */

interface Props {
  state: RunState;
  topology: { nodes: TopologyNode[]; edges: TopologyEdge[] } | null;
}

interface TopologyNode {
  id: string;
  agent: AgentName;
  label: string;
}

interface TopologyEdge {
  from: string;
  to: string;
  kind: "flow" | "reroute";
}

/** Laid out in two rows: discovery along the top, analysis along the bottom. */
const POSITIONS: Record<string, { x: number; y: number }> = {
  scout: { x: 0, y: 0 },
  panel: { x: 175, y: 0 },
  question: { x: 350, y: 0 },
  data: { x: 525, y: 0 },
  design: { x: 525, y: 130 },
  execute: { x: 350, y: 130 },
  uncertainty: { x: 175, y: 130 },
  critic: { x: 0, y: 130 },
  writer: { x: 0, y: 260 },
};

type NodeStatus = "idle" | "active" | "done";

function AgentNode({ data }: { data: { label: string; color: string; status: NodeStatus; short: string } }) {
  const { label, color, status, short } = data;

  const ring =
    status === "active"
      ? "border-2 shadow-lg"
      : status === "done"
        ? "border opacity-90"
        : "border opacity-45";

  return (
    <div
      className={`relative px-3 py-2 rounded-lg bg-[var(--color-panel-2)] ${ring} transition-all duration-300 min-w-[132px]`}
      style={{ borderColor: status === "idle" ? "var(--color-edge)" : color, color }}
    >
      <Handle type="target" position={Position.Left} className="!bg-slate-600 !w-1.5 !h-1.5" />
      <div className="flex items-center gap-2">
        <span
          className={`w-1.5 h-1.5 rounded-full shrink-0 ${status === "active" ? "pulse-ring" : ""}`}
          style={{ background: status === "idle" ? "#475569" : color }}
        />
        <span className="text-[11px] mono tracking-wide">{short}</span>
      </div>
      <div className="text-[11px] text-slate-300 mt-0.5 capitalize leading-tight">{label}</div>
      <Handle type="source" position={Position.Right} className="!bg-slate-600 !w-1.5 !h-1.5" />
    </div>
  );
}

const NODE_TYPES = { agent: AgentNode };

export function LiveGraph({ state, topology }: Props) {
  const [nodes, setNodes, onNodesChange] = useNodesState<Node>([]);
  const [edges, setEdges, onEdgesChange] = useEdgesState<Edge>([]);

  // Reroutes taken this run, so a traversed reroute edge stays highlighted
  // rather than flashing once and vanishing.
  const takenReroutes = useMemo(
    () => new Set(state.reroutes.map((r) => `critic->${r.to}`)),
    [state.reroutes],
  );

  useEffect(() => {
    if (!topology) return;

    setNodes(
      topology.nodes.map((node) => {
        const agent = AGENTS[node.agent];
        const status: NodeStatus = state.activeAgents.has(node.agent)
          ? "active"
          : state.completedAgents.has(node.agent)
            ? "done"
            : "idle";
        return {
          id: node.id,
          type: "agent",
          position: POSITIONS[node.id] ?? { x: 0, y: 0 },
          data: { label: node.label, color: agent.color, status, short: agent.short },
          draggable: false,
        };
      }),
    );

    setEdges(
      topology.edges.map((edge) => {
        const isReroute = edge.kind === "reroute";
        const taken = takenReroutes.has(`${edge.from}->${edge.to}`);
        return {
          id: `${edge.from}-${edge.to}-${edge.kind}`,
          source: edge.from,
          target: edge.to,
          animated: taken,
          style: {
            stroke: isReroute ? (taken ? "#f87171" : "#3f2a3a") : "#2f3a5c",
            strokeWidth: taken ? 2 : 1,
            strokeDasharray: isReroute ? "4 3" : undefined,
          },
          label: taken ? "retry" : undefined,
          labelStyle: { fill: "#f87171", fontSize: 10 },
          labelBgStyle: { fill: "#0e1220" },
        };
      }),
    );
  }, [topology, state.activeAgents, state.completedAgents, takenReroutes, setNodes, setEdges]);

  if (!topology) {
    return (
      <div className="h-full rounded-xl border border-[var(--color-edge)] bg-[var(--color-panel)] flex items-center justify-center">
        <p className="text-sm text-slate-600">Loading graph…</p>
      </div>
    );
  }

  return (
    <div className="h-full rounded-xl border border-[var(--color-edge)] bg-[var(--color-panel)] overflow-hidden relative">
      <div className="absolute top-2 left-3 z-10 flex items-center gap-3 text-[11px] text-slate-500">
        <span className="uppercase tracking-wider">Agent graph</span>
        {state.cycle > 0 && (
          <span className="px-1.5 py-0.5 rounded bg-[var(--color-panel-2)] text-amber-400">
            cycle {state.cycle}/{state.maxCycles}
          </span>
        )}
        {state.reroutes.length > 0 && (
          <span className="text-rose-400">
            {state.reroutes.length} reroute{state.reroutes.length > 1 ? "s" : ""}
          </span>
        )}
      </div>

      <ReactFlow
        nodes={nodes}
        edges={edges}
        onNodesChange={onNodesChange}
        onEdgesChange={onEdgesChange}
        nodeTypes={NODE_TYPES}
        fitView
        fitViewOptions={{ padding: 0.18 }}
        proOptions={{ hideAttribution: true }}
        nodesConnectable={false}
        elementsSelectable={false}
        minZoom={0.4}
        maxZoom={1.4}
      >
        <Background color="#1b2237" gap={18} size={1} />
        <Controls showInteractive={false} className="!bg-[var(--color-panel-2)] !border-[var(--color-edge)]" />
      </ReactFlow>
    </div>
  );
}
