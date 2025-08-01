"""
Graph construction for stock relationships.
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from sklearn.preprocessing import StandardScaler
from src.utils import logger
import networkx as nx
import json

from config.settings import FEATURE_CONFIG, load_stocks_config
from src.utils.bigquery import BigQueryClient


class StockGraphConstructor:
    """Construct graph representation of stock market relationships."""

    def __init__(self, config: Dict = None):
        """Initialize graph constructor."""
        self.config = config or FEATURE_CONFIG['graph_construction']
        self.bq_client = BigQueryClient()
        self.stocks_config = load_stocks_config()

    def build_graph(
            self,
            start_date: str,
            end_date: str,
            min_correlation: float = None
    ) -> Dict[str, Dict[str, float]]:
        """
        Build graph structure based on stock relationships.

        Returns:
            Dictionary of adjacency lists with edge weights
        """
        min_correlation = min_correlation or self.config['min_correlation']

        # Get stock metadata
        metadata = self._get_stock_metadata()

        # Calculate correlations
        correlations = self._calculate_correlations(start_date, end_date)

        # Build graph
        graph = {}

        for ticker1 in metadata.keys():
            graph[ticker1] = {}

            for ticker2 in metadata.keys():
                if ticker1 == ticker2:
                    continue

                # Calculate edge weight based on multiple factors
                weight = self._calculate_edge_weight(
                    ticker1, ticker2, metadata, correlations
                )

                if weight > min_correlation:
                    graph[ticker1][ticker2] = weight

        # Add sector-based connections
        graph = self._add_sector_connections(graph, metadata)

        # Ensure graph is connected
        graph = self._ensure_connected_graph(graph)

        logger.info(f"Built graph with {len(graph)} nodes and "
                    f"{sum(len(edges) for edges in graph.values())} edges")

        return graph

    def _get_stock_metadata(self) -> Dict[str, Dict]:
        """Get stock metadata from configuration and database."""
        metadata = {}

        # From configuration
        for sector, stocks in self.stocks_config.items():
            if sector == 'indices':
                continue
            for stock_info in stocks:
                ticker = stock_info['ticker']
                metadata[ticker] = {
                    'sector': sector,
                    'name': stock_info['name'],
                    'market_cap': stock_info['market_cap'],
                    'sub_industry': stock_info['sub_industry']
                }

        # Fetch additional metadata from BigQuery if available
        query = """
                SELECT ticker, sector, industry, market_cap_category
                FROM `{BQ_TABLES['stock_metadata']}` \
                """

        try:
            df = self.bq_client.query(query)
            for _, row in df.iterrows():
                if row['ticker'] in metadata:
                    metadata[row['ticker']].update({
                        'industry': row['industry'],
                        'market_cap_category': row['market_cap_category']
                    })
        except Exception as e:
            logger.warning(f"Could not fetch metadata from BigQuery: {e}")

        return metadata

    def _calculate_correlations(
            self,
            start_date: str,
            end_date: str
    ) -> pd.DataFrame:
        """Calculate pairwise correlations between stocks."""
        query = f"""
        SELECT 
            ticker,
            date,
            close,
            volume
        FROM `{self.bq_client.dataset_ref}.raw_ohlcv`
        WHERE date BETWEEN '{start_date}' AND '{end_date}'
        ORDER BY ticker, date
        """

        df = self.bq_client.query(query)

        # Calculate returns
        returns_df = df.pivot(index='date', columns='ticker', values='close').pct_change()

        # Calculate correlations
        correlations = returns_df.corr()

        # Also calculate volume correlations
        volume_df = df.pivot(index='date', columns='ticker', values='volume').pct_change()
        volume_corr = volume_df.corr()

        # Store both for edge weight calculation
        self._price_correlations = correlations
        self._volume_correlations = volume_corr

        return correlations

    def _calculate_edge_weight(
            self,
            ticker1: str,
            ticker2: str,
            metadata: Dict,
            correlations: pd.DataFrame
    ) -> float:
        """Calculate edge weight between two stocks."""
        weights = []

        # Price correlation weight
        if ticker1 in correlations.index and ticker2 in correlations.columns:
            price_corr = abs(correlations.loc[ticker1, ticker2])
            weights.append(('correlation', price_corr, self.config['correlation_weight']))

        # Sector similarity weight
        if metadata[ticker1]['sector'] == metadata[ticker2]['sector']:
            sector_weight = 0.8
            if metadata[ticker1].get('sub_industry') == metadata[ticker2].get('sub_industry'):
                sector_weight = 1.0
        else:
            sector_weight = 0.2
        weights.append(('sector', sector_weight, self.config['sector_weight']))

        # Market cap similarity weight
        market_cap_weight = 0.5
        if metadata[ticker1]['market_cap'] == metadata[ticker2]['market_cap']:
            market_cap_weight = 1.0
        weights.append(('market_cap', market_cap_weight, self.config['market_cap_weight']))

        # Calculate weighted sum
        total_weight = sum(value * weight for _, value, weight in weights)

        return total_weight

    def _add_sector_connections(
            self,
            graph: Dict[str, Dict[str, float]],
            metadata: Dict[str, Dict]
    ) -> Dict[str, Dict[str, float]]:
        """Ensure stocks within same sector are connected."""
        sectors = {}

        # Group stocks by sector
        for ticker, meta in metadata.items():
            sector = meta['sector']
            if sector not in sectors:
                sectors[sector] = []
            sectors[sector].append(ticker)

        # Add sector ETF connections
        sector_etfs = {
            'technology': 'QQQ',
            'financials': 'XLF',
            'healthcare': 'XLV',
            'energy': 'XLE',
            'consumer_discretionary': 'XLY',
            'consumer_staples': 'XLP',
            'industrials': 'XLI',
            'utilities': 'XLU',
            'materials': 'XLB',
            'real_estate': 'XLRE'
        }

        # Connect stocks to sector ETFs and ensure minimum sector connectivity
        for sector, tickers in sectors.items():
            etf = sector_etfs.get(sector)

            # Find sector hub (most connected stock in sector)
            sector_connections = {
                ticker: len(graph.get(ticker, {}))
                for ticker in tickers
            }
            hub_ticker = max(sector_connections, key=sector_connections.get)

            for ticker in tickers:
                # Connect to sector ETF if available
                if etf and etf in graph:
                    if etf not in graph[ticker]:
                        graph[ticker][etf] = 0.5
                    if ticker not in graph[etf]:
                        graph[etf][ticker] = 0.5

                # Ensure connection to sector hub
                if ticker != hub_ticker:
                    if hub_ticker not in graph[ticker]:
                        graph[ticker][hub_ticker] = 0.3
                    if ticker not in graph[hub_ticker]:
                        graph[hub_ticker][ticker] = 0.3

        return graph

    def _ensure_connected_graph(
            self,
            graph: Dict[str, Dict[str, float]]
    ) -> Dict[str, Dict[str, float]]:
        """Ensure the graph is fully connected."""
        # Convert to NetworkX for connectivity analysis
        G = nx.Graph()
        for source, targets in graph.items():
            for target, weight in targets.items():
                G.add_edge(source, target, weight=weight)

        # Find connected components
        components = list(nx.connected_components(G))

        if len(components) > 1:
            logger.warning(f"Graph has {len(components)} disconnected components")

            # Connect components through market index (SPY)
            spy = 'SPY'
            for component in components:
                # Find node with highest degree in component
                component_hub = max(
                    component,
                    key=lambda n: G.degree(n) if n in G else 0
                )

                # Connect to SPY
                if spy not in graph:
                    graph[spy] = {}
                if component_hub not in graph[spy]:
                    graph[spy][component_hub] = 0.4
                if spy not in graph[component_hub]:
                    graph[component_hub] = {}
                    graph[component_hub][spy] = 0.4

        return graph

    def calculate_graph_features(
            self,
            graph: Dict[str, Dict[str, float]]
    ) -> pd.DataFrame:
        """Calculate graph-based features for each node."""
        # Convert to NetworkX
        G = nx.Graph()
        for source, targets in graph.items():
            for target, weight in targets.items():
                G.add_edge(source, target, weight=weight)

        features = []

        for node in G.nodes():
            node_features = {
                'ticker': node,
                'degree': G.degree(node),
                'weighted_degree': G.degree(node, weight='weight'),
                'clustering_coefficient': nx.clustering(G, node),
                'betweenness_centrality': nx.betweenness_centrality(G)[node],
                'closeness_centrality': nx.closeness_centrality(G)[node],
                'eigenvector_centrality': nx.eigenvector_centrality(G, max_iter=1000)[node],
            }

            # PageRank
            pagerank = nx.pagerank(G, weight='weight')
            node_features['pagerank'] = pagerank[node]

            # Average neighbor degree
            avg_neighbor_degree = nx.average_neighbor_degree(G, weight='weight')
            node_features['avg_neighbor_degree'] = avg_neighbor_degree[node]

            features.append(node_features)

        return pd.DataFrame(features)

    def save_graph(self, graph: Dict[str, Dict[str, float]], filename: str):
        """Save graph structure to file."""
        with open(filename, 'w') as f:
            json.dump(graph, f, indent=2)
        logger.info(f"Saved graph to {filename}")

    def load_graph(self, filename: str) -> Dict[str, Dict[str, float]]:
        """Load graph structure from file."""
        with open(filename, 'r') as f:
            graph = json.load(f)
        logger.info(f"Loaded graph from {filename}")
        return graph

    def visualize_graph(
            self,
            graph: Dict[str, Dict[str, float]],
            output_file: str = 'stock_graph.png',
            show_labels: bool = True
    ):
        """Create visualization of the stock graph."""
        import matplotlib.pyplot as plt

        # Convert to NetworkX
        G = nx.Graph()
        for source, targets in graph.items():
            for target, weight in targets.items():
                G.add_edge(source, target, weight=weight)

        # Get metadata for coloring
        metadata = self._get_stock_metadata()

        # Create color map by sector
        sectors = list(set(meta['sector'] for meta in metadata.values()))
        color_map = plt.cm.get_cmap('tab20')
        sector_colors = {sector: color_map(i/len(sectors))
                         for i, sector in enumerate(sectors)}

        # Node colors
        node_colors = []
        for node in G.nodes():
            if node in metadata:
                sector = metadata[node]['sector']
                node_colors.append(sector_colors[sector])
            else:
                node_colors.append('gray')

        # Layout
        pos = nx.spring_layout(G, k=2, iterations=50)

        # Draw
        plt.figure(figsize=(20, 16))

        # Draw edges with varying thickness based on weight
        edges = G.edges()
        weights = [G[u][v]['weight'] for u, v in edges]

        nx.draw_networkx_edges(
            G, pos,
            width=[w * 2 for w in weights],
            alpha=0.3,
            edge_color='gray'
        )

        # Draw nodes
        node_sizes = [G.degree(node) * 100 for node in G.nodes()]
        nx.draw_networkx_nodes(
            G, pos,
            node_color=node_colors,
            node_size=node_sizes,
            alpha=0.8
        )

        if show_labels:
            nx.draw_networkx_labels(
                G, pos,
                font_size=8,
                font_weight='bold'
            )

        # Legend
        for sector, color in sector_colors.items():
            plt.scatter([], [], c=[color], label=sector, s=100)
        plt.legend(loc='upper left', bbox_to_anchor=(1, 1))

        plt.title("Stock Market Graph Structure", fontsize=16)
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()

        logger.info(f"Saved graph visualization to {output_file}")

    def get_subgraph(
            self,
            graph: Dict[str, Dict[str, float]],
            center_ticker: str,
            max_distance: int = 2
    ) -> Dict[str, Dict[str, float]]:
        """Extract subgraph around a specific ticker."""
        # Convert to NetworkX
        G = nx.Graph()
        for source, targets in graph.items():
            for target, weight in targets.items():
                G.add_edge(source, target, weight=weight)

        # Find nodes within distance
        subgraph_nodes = set([center_ticker])
        current_layer = set([center_ticker])

        for _ in range(max_distance):
            next_layer = set()
            for node in current_layer:
                if node in G:
                    next_layer.update(G.neighbors(node))
            subgraph_nodes.update(next_layer)
            current_layer = next_layer

        # Build subgraph
        subgraph = {}
        for node in subgraph_nodes:
            if node in graph:
                subgraph[node] = {
                    target: weight
                    for target, weight in graph[node].items()
                    if target in subgraph_nodes
                }

        return subgraph