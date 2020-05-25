import osmnx as ox
import networkx as nx
import queue
import math
import priority_dict

def get_path(origin_key, goal_key, predecessors):
    key = goal_key
    path = [goal_key]

    while(key != origin_key):
        key = predecessors[key]
        path.insert(0, key)

    return path

def dijkstras_search(origin_key, goal_key, graph):
    open_queue = priority_dict.priority_dict({})
    open_queue[origin_key] = 0.0
    closed_dict = {}
    predecessors = {}
    goal_found = False

    while(open_queue):
        u, u_length = open_queue.pop_smallest()

        if u == goal_key:
            goal_found = True
            break
        for edge_dict in graph.out_edges([u], data=True):
            v = edge_dict[1]
            if v in closed_dict:
                continue
            uv_length = edge_dict[2]['length']

            if v not in open_queue:
                open_queue[v] = u_length + uv_length
                predecessors[v] = u
            else:
                v_length = open_queue[v]
                if u_length + uv_length < v_length:
                    open_queue[v] = u_length + uv_length
                    predecessors[v] = u

        closed_dict[u] = 1

    if not goal_found:
        raise ValueError("Goal not found in search.")

    return get_path(origin_key, goal_key, predecessors)


def distance_heuristic(state_key, goal_key, node_data):
    n1 = node_data[state_key]
    n2 = node_data[goal_key]

    long1 = n1['x'] * math.pi / 180.0
    lat1 = n1['y'] * math.pi / 180.0
    long2 = n2['x'] * math.pi / 180.0
    lat2 = n2['y'] * math.pi / 180.0

    r = 6371000

    x1 = r * math.cos(lat1) * math.cos(long1)
    y1 = r * math.cos(lat1) * math.sin(long1)
    z1 = r*math.sin(lat1)

    x2 = r * math.cos(lat2) * math.cos(long2)
    y2 = r * math.cos(lat2) * math.sin(long2)
    z2 = r * math.sin(lat2)

    d = ((x2 - x1)**2 + (y2 - y1)**2 + (z2 - z1)**2)**0.5

    return d

def a_star_search(origin_key, goal_key, graph):
    node_data = graph.nodes(True)

    open_queue = priority_dict.priority_dict({})
    open_queue[origin_key] = distance_heuristic(origin_key, goal_key, node_data)

    closed_dict = {}
    predecessors = {}

    costs = {}
    costs[origin_key] = 0.0

    goal_found = False

    while open_queue:
        u, u_heuristic = open_queue.pop_smallest()
        u_length = costs[u]
        if u == goal_key:
            goal_found = True
            break

        for edge_dict in graph.out_edges([u], data=True):
            v = edge_dict[1]

            if v in closed_dict:
                continue

            uv_length = edge_dict[2]['length']

            if v not in open_queue:
                costs[v] = u_length + uv_length
                open_queue[v] = u_length + uv_length + distance_heuristic(v, goal_key, node_data)
                predecessors[v] = u
            else:
                v_length = costs[v]
                if u_length + uv_length < v_length:
                    costs[v] = u_length + uv_length
                    open_queue[v] = u_length + uv_length + distance_heuristic(v, goal_key, node_data)
                    predecessors[v] = u

        closed_dict[u] = 1

    if not goal_found:
        raise ValueError("Goal not found in search.")

    return get_path(origin_key, goal_key, predecessors)




def main():
    map_graph = ox.graph_from_place('Berkley, California', network_type='drive')
    origin = ox.get_nearest_node(map_graph, (37.8743, -122.277))
    destination = list(map_graph.nodes())[-1]

    shortest_path = nx.shortest_path(map_graph, origin, destination, weight='length')
    fig, ax = ox.plot_graph_route(map_graph, shortest_path)

    path = dijkstras_search(origin, destination, map_graph)
    fig, ax = ox.plot_graph_route(map_graph, path)

    path = a_star_search(origin, destination, map_graph)
    fig, ax = ox.plot_graph_route(map_graph, path)

if __name__ == "__name__":
    main()