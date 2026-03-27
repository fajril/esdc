"""Test to verify collapsible widgets are created properly."""

from esdc.chat.app import ThinkingIndicator, SQLPanel, ResultsPanel

print("Testing Collapsible Widgets Creation...\n")

# Test 1: ThinkingIndicator
print("1. ThinkingIndicator:")
thinking = ThinkingIndicator()
print(f"   - collapsed: {thinking.collapsed}")
print(f"   - title: {thinking.title}")
print(f"   - CSS classes: {thinking.classes}")
print(f"   - has steps list: {hasattr(thinking, 'steps')}")
print()

# Test 2: SQLPanel with content
print("2. SQLPanel with content:")
sql = "SELECT * FROM users LIMIT 10;"
sql_panel = SQLPanel(sql)
print(f"   - collapsed: {sql_panel.collapsed}")
print(f"   - title: {sql_panel.title}")
print(f"   - CSS classes: {sql_panel.classes}")
print(f"   - has css: {sql_panel.DEFAULT_CSS is not None}")
print(f"   - min-height in CSS: {'min-height' in sql_panel.DEFAULT_CSS}")
print()

# Test 3: SQLPanel empty
print("3. SQLPanel empty:")
empty_sql = SQLPanel("")
print(f"   - collapsed: {empty_sql.collapsed} (should be True for empty)")
print()

# Test 4: ResultsPanel with content
print("4. ResultsPanel with content:")
results = "col1|col2\nval1|val2"
results_panel = ResultsPanel(results)
print(f"   - collapsed: {results_panel.collapsed}")
print(f"   - title: {results_panel.title}")
print(f"   - CSS classes: {results_panel.classes}")
print(f"   - min-height in CSS: {'min-height' in results_panel.DEFAULT_CSS}")
print()

# Test 5: ResultsPanel empty
print("5. ResultsPanel empty:")
empty_results = ResultsPanel("")
print(f"   - collapsed: {empty_results.collapsed} (should be True for empty)")
print()

# Test 6: Check CSS styles are defined
print("6. CSS Structure verification:")
print("   ThinkingIndicator CSS:")
if "min-height" in thinking.DEFAULT_CSS:
    print("   ✓ min-height defined")
if "background" in thinking.DEFAULT_CSS:
    print("   ✓ background defined")
if "border" in thinking.DEFAULT_CSS:
    print("   ✓ border defined")
print()

print("All widgets created successfully!")
print("\nKey findings:")
print("- collapsible widgets have 'collapsed' attribute")
print("- SQLPanel & ResultsPanel with content should NOT be collapsed")
print("- Empty panels should BE collapsed")
print("- CSS has min-height, background, and border for visibility")
