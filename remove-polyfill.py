import os
import re
from bs4 import BeautifulSoup

# Define the polyfill.io domains to look for
POLYFILL_DOMAINS = [
    'polyfill.io',
    'cdn.polyfill.io',
    'www.polyfill.io'
]

def remove_polyfill_links(html_content):
    """Remove script tags that load from polyfill.io domains."""
    soup = BeautifulSoup(html_content, 'html.parser')
    
    # Find all script tags
    scripts = soup.find_all('script')
    
    removed_count = 0
    for script in scripts:
        src = script.get('src')
        if src:
            # Check if the src attribute contains any of the polyfill domains
            for domain in POLYFILL_DOMAINS:
                if domain in src:
                    # Remove the entire script tag
                    script.decompose()
                    removed_count += 1
                    print(f"Removed script with src: {src}")
                    break  # Stop checking other domains for this script
    
    # Return the modified HTML as a string
    return str(soup), removed_count

def process_file(filepath):
    """Process a single HTML file."""
    try:
        with open(filepath, 'r', encoding='utf-8') as file:
            content = file.read()
        
        # Only process if the file contains 'polyfill.io' (quick check to save time)
        if 'polyfill.io' not in content:
            return 0
        
        new_content, count = remove_polyfill_links(content)
        
        if count > 0:
            # Write the changes back to the file
            with open(filepath, 'w', encoding='utf-8') as file:
                file.write(new_content)
            print(f"✅ Updated: {filepath} (removed {count} script(s))")
        
        return count
    except Exception as e:
        print(f"❌ Error processing {filepath}: {e}")
        return 0

def main():
    """Walk through the project directory and process all HTML files."""
    total_removed = 0
    html_extensions = ('.html', '.htm')
    
    for root, dirs, files in os.walk('.'):
        # Skip hidden directories like .git
        if '.git' in root:
            continue
            
        for file in files:
            if file.endswith(html_extensions):
                filepath = os.path.join(root, file)
                total_removed += process_file(filepath)
    
    print(f"\n✨ Complete! Total scripts removed: {total_removed}")

if __name__ == "__main__":
    main()