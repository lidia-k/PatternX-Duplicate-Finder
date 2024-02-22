from Duplicate_Finder.duplicate_finder import DuplicateFinder

if __name__ == '__main__':
    TABLE_NAME = 'production.product'
    duplicate_finder = DuplicateFinder('sentence-transformers/all-MiniLM-L6-v2', TABLE_NAME)
    #duplicate_finder.extract_lowest_distances(English=True)

    size_prompt = """The product subcategory is the most important feature to consider.
    The difference in size is less important than the product subcategory.
    """
    #For example, the size 42 is very different from 48."""
    example1 = [964, 965, 961]
    example2 = [765, 766, 768]

    duplicate_finder.extract_distance_between_pairs([765, 10001, 10002], prompt=size_prompt, English=True)