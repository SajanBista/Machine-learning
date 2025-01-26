from imdb import IMDb
def get_movie_rating ():
    movie_name = input( "Enter the name of the movie: ")
    ia = IMDb
    movies = ia.search_movie(movie_name)


    if movies:
        movie = movies[0]
        ia.update(movie)
        rating = movie.get('rating', 'N/A')
        print(f"The IMDb rating of '{movie_name}' is: {rating}")
    else:
        print(f"Movie '{movie_name}' not found!")
get_movie_rating()