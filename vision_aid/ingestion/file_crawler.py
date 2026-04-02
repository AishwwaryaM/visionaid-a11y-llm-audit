import requests
from datetime import datetime
from typing import Optional, List, Set
from urllib.parse import urljoin, urlparse
import re
import time

def download_html(url: str, filename: Optional[str] = None, depth: int = 1, 
                  visited: Optional[Set[str]] = None, base_domain: Optional[str] = None) -> List[str]:
    """
    Download HTML from a URL and recursively follow links to nested pages.
    
    Args:
        url: The starting URL to download
        filename: Optional base filename (will be appended with _n for nested pages)
        depth: How many levels deep to crawl (default: 1)
        visited: Set of already visited URLs (used internally for recursion)
        base_domain: The base domain to stay within (used internally)
    
    Returns:
        List of saved filenames
    """
    if visited is None:
        visited = set()
    
    if base_domain is None:
        # Extract base domain from the starting URL
        parsed_url = urlparse(url)
        base_domain = parsed_url.netloc
    
    saved_files = []
    
    # Skip if already visited or depth is negative
    if url in visited or depth < 0:
        return saved_files
    
    visited.add(url)
    
    try:
        print(f"Downloading: {url} (depth: {depth})")
        response = requests.get(url, headers={'User-Agent': 'Mozilla/5.0'})
        response.raise_for_status()
        
        # Generate filename
        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            domain = url.replace('https://', '').replace('http://', '').split('/')[0]
            
            # Create unique filename based on URL path
            path_part = urlparse(url).path.replace('/', '_').strip('_')
            if path_part:
                filename = f"{domain}_{path_part}_{timestamp}.html"
            else:
                filename = f"{domain}_index_{timestamp}.html"
        else:
            # If base filename provided, append depth and index
            base_name = filename.replace('.html', '')
            filename = f"{base_name}_depth{depth}_{len(saved_files)}.html"
        
        # Save the HTML
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(response.text)
        
        print(f"Saved: {filename}")
        saved_files.append(filename)
        
        # Recursively download nested pages if depth > 0
        if depth > 0:
            # Find all links in the page
            links = extract_links(response.text, url)
            
            # Filter links to stay within the same domain
            nested_urls = []
            for link in links:
                parsed_link = urlparse(link)
                if parsed_link.netloc == base_domain or not parsed_link.netloc:
                    # Normalize relative URLs
                    absolute_url = urljoin(url, link)
                    if absolute_url not in visited:
                        nested_urls.append(absolute_url)
            
            # Download nested pages
            for nested_url in nested_urls[:10]:  # Limit to 10 links per page
                time.sleep(0.5)
                nested_files = download_html(
                    nested_url, 
                    filename=None,
                    depth=depth - 1,
                    visited=visited,
                    base_domain=base_domain
                )
                saved_files.extend(nested_files)
        
        return saved_files
        
    except Exception as e:
        print(f"Error downloading {url}: {e}")
        return saved_files

def extract_links(html_content: str, base_url: str) -> Set[str]:
    """Extract all href links from HTML content."""
    links = set()
    
    # Simple regex to find href attributes
    href_pattern = r'href=["\'](.*?)["\']'
    matches = re.findall(href_pattern, html_content, re.IGNORECASE)
    
    # Extensions and path patterns that are not HTML pages
    _SKIP_EXT = (
        '.css', '.js', '.json', '.xml', '.rss', '.atom',
        '.pdf', '.jpg', '.jpeg', '.png', '.gif', '.svg', '.webp',
        '.zip', '.gz', '.tar', '.woff', '.woff2', '.ttf', '.eot',
    )
    _SKIP_PATH = ('/feed/', '/feed', '/wp-json/', '/xmlrpc.php', '/comments/feed')

    for match in matches:
        # Skip empty links, javascript, mailto, etc.
        if (match and not match.startswith('#') and
            not match.startswith('javascript:') and
            not match.startswith('mailto:')):

            # Convert to absolute URL
            absolute_url = urljoin(base_url, match)

            # Only include http(s) URLs
            if not absolute_url.startswith(('http://', 'https://')):
                continue

            # Skip non-page resources
            parsed_path = urlparse(absolute_url).path.lower()
            query = urlparse(absolute_url).query
            if any(parsed_path.endswith(ext) for ext in _SKIP_EXT):
                continue
            if any(seg in parsed_path for seg in _SKIP_PATH):
                continue
            if 'wp-json' in query or 'wp-content' in parsed_path:
                continue

            links.add(absolute_url)
    
    return links

def fetch_page(url: str, timeout: int = 30) -> str:
    """Fetch a single URL and return its HTML content as a string."""
    print(f"[fetch_page] GET {url}")
    response = requests.get(url, headers={'User-Agent': 'Mozilla/5.0'}, timeout=timeout)
    print(f"[fetch_page] Status: {response.status_code}")
    print(f"[fetch_page] Content-Type: {response.headers.get('Content-Type', 'unknown')}")
    print(f"[fetch_page] Content-Length (bytes): {len(response.content):,}")
    print(f"[fetch_page] Final URL (after redirects): {response.url}")
    if response.history:
        print(f"[fetch_page] Redirects: {[r.status_code for r in response.history]}")
    response.raise_for_status()
    html = response.text
    print(f"[fetch_page] HTML chars: {len(html):,}")
    print(f"[fetch_page] First 500 chars:\n{html[:500]}")
    return html


def _normalize_url(url: str) -> str:
    """Normalize a URL so that trailing-slash variants are treated as identical."""
    parsed = urlparse(url)
    path = parsed.path
    if not path or path == "/":
        path = "/"
    elif not path.endswith("/") and "." not in path.rsplit("/", 1)[-1]:
        # Add trailing slash to directory-like paths (no file extension)
        path += "/"
    return parsed._replace(path=path, fragment="").geturl()


def fetch_pages_nested(url: str, max_depth: int = 1,
                       max_links_per_page: int = 10, timeout: int = 30) -> tuple:
    """Fetch HTML from *url* and its in-domain links up to *max_depth* levels.

    All pages are concatenated with HTML comment separators.

    Returns:
        Tuple of (html_str, tree) where *tree* maps each crawled URL to the
        list of direct child URLs discovered from that page.
    """
    url = _normalize_url(url)
    base_domain = urlparse(url).netloc
    visited: Set[str] = set()
    parts: List[str] = []
    tree: dict = {}  # parent_url -> [child_urls]

    def _crawl(current_url: str, depth: int, parent_url: str = None) -> None:
        current_url = _normalize_url(current_url)
        if current_url in visited or depth < 0:
            return
        visited.add(current_url)

        # Register in tree
        if parent_url is not None:
            tree.setdefault(parent_url, []).append(current_url)
        tree.setdefault(current_url, [])  # ensure node exists even if leaf

        try:
            resp = requests.get(current_url, headers={'User-Agent': 'Mozilla/5.0'},
                                timeout=timeout)
            resp.raise_for_status()
            parts.append(f"<!-- PAGE: {current_url} -->\n{resp.text}")
            if depth > 0:
                links = extract_links(resp.text, current_url)
                nested = [
                    link for link in links
                    if urlparse(link).netloc == base_domain and link not in visited
                ]
                for link in nested[:max_links_per_page]:
                    time.sleep(0.5)
                    _crawl(link, depth - 1, parent_url=current_url)
        except Exception as e:
            print(f"Warning: could not fetch {current_url}: {e}")

    _crawl(url, max_depth)
    return "\n\n".join(parts), tree


if __name__ == "__main__":
    # Download a page and its nested links up to depth 1
    start_url = "https://visionaid.org/"  # Replace with your target URL
    
    print(f"Starting download from {start_url}")
    saved_files = download_html(start_url, depth=1)
    
    print(f"\nDownloaded {len(saved_files)} files:")
    for file in saved_files:
        print(f"  - {file}")