import tables

h5f = tables.open_file('db_csi.h5', 'a')

h5f.root.ismir_artist.cols.date.remove_index()
h5f.root.ismir_artist.cols.date.create_csindex()

h5f.root.ismir_song.cols.date.remove_index()
h5f.root.ismir_song.cols.date.create_csindex()

h5f.root.lastfm_artist.cols.date.remove_index()
h5f.root.lastfm_artist.cols.date.create_csindex()

h5f.root.lastfm_song.cols.date.remove_index()
h5f.root.lastfm_song.cols.date.create_csindex()

h5f.root.twitter_hashtags.cols.date.remove_index()
h5f.root.twitter_hashtags.cols.date.create_csindex()

h5f.close()
