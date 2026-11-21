"""photosweep -- find screenshots, memes and other throwaway images in one
Nextcloud photo folder, and review them for deletion from a tailnet page.

The scanner labels; a person deletes. Nothing in this package removes a file
from disk: a delete is a WebDAV DELETE against Nextcloud, which moves the file
to that user's trash bin.
"""
