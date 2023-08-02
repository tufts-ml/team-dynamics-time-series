import glob
import os
from typing import List, Optional

import prettyprinter as pp


pp.install_extras()

from dynagroup.model2a.basketball.data.baller2vec.core import BasketballGame
from dynagroup.model2a.basketball.data.baller2vec.get_game import get_basketball_game


"""
The CLE dataset consists of 29 games available from the NBA 2015-2016 repo 
at https://github.com/linouk23/NBA-Player-Movements
and containing one of 4 common starting lineups from CLE.

The datasets used here are listed in 
/Users/mwojno01/Repos/baller2vec_forked/src/baller2vec_forked/mike/CLE_starters.py
"""


def get_basketball_games_for_CLE_dataset(
    event_idxs: Optional[List[int]] = None,
    sampling_rate_Hz: int = 5,
    verbose: bool = True,
) -> List[BasketballGame]:
    PATH_TO_BALLER2VEC_INFO = "/Users/mwojno01/Repos/dynagroup/data/basketball/baller2vec_format/CLE_starters/info/baller2vec_info.pydict"

    GAME_AND_EVENT_LABEL_DATA_DIR = (
        "/Users/mwojno01/Repos/dynagroup/data/basketball/baller2vec_format/CLE_starters/games/"
    )
    # Get a list of all files in the directory that end with *_X.npy
    X_paths = glob.glob(os.path.join(GAME_AND_EVENT_LABEL_DATA_DIR, "*_X.npy"))
    y_paths = glob.glob(os.path.join(GAME_AND_EVENT_LABEL_DATA_DIR, "*_y.npy"))

    n_games = len(X_paths)
    games = [None] * n_games

    for g, (X_path, y_path) in enumerate(zip(X_paths, y_paths)):
        if verbose:
            print(f"Now processing game {g+1}/{n_games} from CLE starters dataset.")
        games[g] = get_basketball_game_with_CLE_starters(
            X_path,
            y_path,
            PATH_TO_BALLER2VEC_INFO,
            event_idxs,
            sampling_rate_Hz,
        )

    # TODO: Move the below summary stats OUTSIDE of the CLE function.
    # That would be a better location, e.g. later we will want to know the number of plays in the
    # train, test, and validation set.  We will also want to compute the total number of timesteps.
    plays_per_game = [len(games[g].events) for g in range(n_games)]
    print(f"The number of plays (with CLE starters) per game is {plays_per_game}")
    print(f"The total number of plays is {sum(plays_per_game)}.")
    return games


def get_basketball_game_with_CLE_starters(
    path_to_game_data: str,
    path_to_event_label_data: str,
    path_to_baller2vec_info: str,
    event_idxs: Optional[List[int]] = None,
    sampling_rate_Hz: int = 5,
) -> BasketballGame:
    """
    We only keep events with the starters for a given game.

    An incomplete list of steps along the way:
        1. Filter out all plays where we don't have the CLE starters.
        2. Run a preprocessing step to assign all available players in the game
        to position idxs.  The CLE starters are given indices [0,1,2,3,4], and the
        opponents are given indices [5,6,7,8,9] by mapping their positions to an index.
        3. The baller2vec `Event`s are extracted and reindexed according to the design above.
        4. We normalize the court so that the focal team (CLE) always has its basket on the left.
            (By "its basket", we mean that hoop_sides=0 in the baller2vec Event representation.
            To ascertain confidently whether this refers to the focal team's offensive or defensive
            hoop would require digging through the docs of both baller2vec and the original dataset.)

            NOTE: I assume that the center of the [0,100]x[0,50] court is [50,25].  But some code in
            the baller2vec repo suggests that the center on the x-axis might be 47 rather than 50
            (e.g. see https://github.com/airalcorn2/baller2vec/blob/master/settings.py#L17).
            Check into this.

    Returns:
        unnormalized coordinates for basketball players,
            array of shape (T_slice, J=10, D=2)
    """

    CLE_POTENTIAL_STARTER_NAMES_2_ENTITY_IDXS = {
        "LeBron James": 0,  # evertyhing
        "Kevin Love": 1,  # power forward, center
        "Timofey Mozgov": 2,  # center
        "Tristan Thompson": 2,  # center
        "J.R. Smith": 3,  # shooting guard/small forward
        "Kyrie Irving": 4,  # guard
        "Mo Williams": 4,  # guard
        "Matthew Dellavedova": 4,  # guard
    }
    # CLE starters were:
    # \texttt{0: Lebron James, 1: Kevin Love, 2: J.R. Smith, 3: Starting Center, 4: Starting Guard}.
    # Depending on the game, the starting center was either T. Mazgov or T. Thompson.
    # Similarly, the starting guard was either K. Irving, M. Williams, or M. Dellavedova.

    ### Identify positions of all players in game

    # Rk: I have the output hardcoded in here so we don't have to deal with Wikipedia scraping
    # each time we load the dataset. The scraping is (1) slow and (2) can sometimes miss positions
    # e.g., consider that some players may have the same name as different person with a Wiki entry,
    # in which case our guess at the Wiki url won't work.   In particular (2) happens for
    # 'Chris Johnson', "James Jones", 'Reggie Jackson',  'Ryan Anderson'
    # 'Jason Smith', 'Kevin Martin',  'Marcus Thornton'.
    # In addition, (3) I manually replaced "Centre" with "Center" for two of the players.
    # and (4) I manually ensured that posiitons after a "/" were lowercase, which affected 2 players.

    # TODO: Deal with "centre/power forward" and other anomalies

    PLAYER_NAMES_IN_DATASET_2_POSITIONS = {
        "Pau Gasol": "Power forward / center",
        "Kirk Hinrich": "Point guard / shooting guard",
        "Joakim Noah": "Center",
        "Aaron Brooks": "Point guard",
        "Derrick Rose": "Point guard",
        "Taj Gibson": "Power forward / center",
        "Nikola Mirotic": "Power forward",
        "Jimmy Butler": "Small forward",
        "E'Twaun Moore": "Shooting guard",
        "Tony Snell": "Shooting guard / small forward",
        "Doug McDermott": "Small forward",
        "Cameron Bairstow": "Power forward / center",
        "Bobby Portis": "Center / power forward",
        "Richard Jefferson": "Small forward",
        "LeBron James": "Small forward / power forward",
        "Mo Williams": "Point guard",
        "James Jones": "Small forward / shooting guard",
        "J.R. Smith": "Shooting guard / small forward",
        "Anderson Varejao": "Center / power forward",
        "Kevin Love": "Power forward",
        "Sasha Kaun": "Center",
        "Timofey Mozgov": "Center",
        "Tristan Thompson": "Center / power forward",
        "Jared Cunningham": "Shooting guard",
        "Matthew Dellavedova": "Point guard / shooting guard",
        "Joe Harris": "Small forward / shooting guard",
        "Vince Carter": "Shooting guard / small forward",
        "Zach Randolph": "Power forward / center",
        "Matt Barnes": "Small forward",
        "Tony Allen": "Shooting guard / small forward",
        "Beno Udrih": "Point guard / shooting guard",
        "Mike Conley": "Point guard",
        "Jeff Green": "Power forward / small forward",
        "Brandan Wright": "Power forward / center",
        "Marc Gasol": "Center",
        "Courtney Lee": "Shooting guard",
        "JaMychal Green": "Power forward / center",
        "Russ Smith": "Point guard",
        "Jordan Adams": "Shooting guard / small forward",
        "Amar'e Stoudemire": "Power forward / center",
        "Chris Bosh": "Power forward / center",
        "Dwyane Wade": "Shooting guard / point guard",
        "Udonis Haslem": "Center / power forward",
        "Luol Deng": "Small forward / power forward",
        "Gerald Green": "Shooting guard / small forward",
        "Josh McRoberts": "Power forward",
        "Mario Chalmers": "Point guard",
        "Goran Dragic": "Point guard",
        "Hassan Whiteside": "Center",
        "James Ennis": "Small forward",
        "Tyler Johnson": "Shooting guard / point guard",
        "Justise Winslow": "Small forward / point guard",
        "Kendall Marshall": "Point guard",
        "Hollis Thompson": "Shooting guard / small forward",
        "Nerlens Noel": "Center",
        "Isaiah Canaan": "Shooting guard",
        "Robert Covington": "Power forward / center",
        "Nik Stauskas": "Shooting guard",
        "Jerami Grant": "Power forward",
        "JaKarr Sampson": "Power forward",
        "TJ McConnell": "Point guard",
        "Jahlil Okafor": "Center",
        "Richaun Holmes": "Power forward / center",
        "Christian Wood": "Center / power forward",
        "Joe Ingles": "Small forward / shooting guard",
        "Trey Burke": "Point guard",
        "Rodney Hood": "Small forward / shooting guard",
        "Alec Burks": "Shooting guard / small forward",
        "Elijah Millsap": "Shooting guard / small forward",
        "Derrick Favors": "Center / power forward",
        "Gordon Hayward": "Small forward / power forward",
        "Tibor Pleiss": "Center",
        "Chris Johnson": "Small forward / power forward",
        "Raul Neto": "Point guard",
        "Rudy Gobert": "Center",
        "Trevor Booker": "Power forward",
        "Trey Lyles": "Power forward / center",
        "Jerian Grant": "Point guard",
        "Langston Galloway": "Shooting guard / point guard",
        "Kevin Seraphin": "Center",
        "Lance Thomas": "Power forward / small forward",
        "Jose Calderon": "Point guard",
        "Kyle O'Quinn": "Center / power forward",
        "Sasha Vujacic": "Shooting guard",
        "Carmelo Anthony": "Small forward / power forward",
        "Kristaps Porzingis": "Center / power forward",
        "Arron Afflalo": "Shooting guard / small forward",
        "Robin Lopez": "Center",
        "Lou Amundson": "Power forward / center",
        "Derrick Williams": "Power forward",
        "Michael Carter-Williams": "Point guard",
        "Chris Copeland": "Small forward / power forward",
        "Jabari Parker": "Power forward",
        "Greg Monroe": "Power forward / center",
        "Damien Inglis": "Power forward / small forward",
        "Miles Plumlee": "Center / power forward",
        "Jerryd Bayless": "Point guard / shooting guard",
        "Rashad Vaughn": "Shooting guard",
        "Greivis Vasquez": "Point guard / shooting guard",
        "Khris Middleton": "Small forward",
        "John Henson": "Power forward / center",
        "Giannis Antetokounmpo": "Power forward / small forward",
        "Johnny O'Bryant": "Center / power forward",
        "Steve Blake": "Point guard",
        "Ersan Ilyasova": "Power forward",
        "Joel Anthony": "Center / power forward",
        "Anthony Tolliver": "Power forward",
        "Marcus Morris": "Power forward",
        "Reggie Jackson": "Point guard / shooting guard",
        "Andre Drummond": "Center",
        "Aron Baynes": "Center / power forward",
        "Kentavious Caldwell-Pope": "Shooting guard",
        "Reggie Bullock": "Small forward / shooting guard",
        "Spencer Dinwiddie": "Point guard / shooting guard",
        "Stanley Johnson": "Small forward / power forward",
        "Darrun Hilliard": "Shooting guard / small forward",
        "O.J. Mayo": "Shooting guard",
        "Spencer Hawes": "Center / power forward",
        "Marvin Williams": "Power forward / small forward",
        "Jeremy Lamb": "Shooting guard / small forward",
        "Nicolas Batum": "Power forward / small forward",
        "Jeremy Lin": "Point guard / shooting guard",
        "Kemba Walker": "Point guard",
        "PJ Hairston": "Small forward / shooting guard",
        "Brian Roberts": "Point guard",
        "Al Jefferson": "Power forward / center",
        "Troy Daniels": "Shooting guard",
        "Cody Zeller": "Center / power forward",
        "Frank Kaminsky": "Center / power forward",
        "Tyler Hansbrough": "Power forward / center",
        "Jared Dudley": "Small forward / power forward",
        "John Wall": "Point guard",
        "Bradley Beal": "Shooting guard",
        "Ramon Sessions": "Point guard",
        "Kelly Oubre": "Small forward",
        "Marcin Gortat": "Center",
        "Garrett Temple": "Shooting guard / small forward",
        "Ryan Hollins": "Center / power forward",
        "Otto Porter": "Small forward / power forward",
        "Kris Humphries": "Power forward",
        "DeJuan Blair": "Power forward / center",
        "Gary Neal": "Shooting guard",
        "Drew Gooden": "Power forward",
        "Tyreke Evans": "Small forward / shooting guard",
        "Omer Asik": "Center",
        "Ish Smith": "Point guard",
        "Luke Babbitt": "Small forward / power forward",
        "Eric Gordon": "Combo guard",
        "Jrue Holiday": "Point guard / shooting guard",
        "Alonzo Gee": "Shooting guard / small forward",
        "Toney Douglas": "Point guard",
        "Anthony Davis": "Center / power forward",
        "Norris Cole": "Point guard",
        "Ryan Anderson": "Power forward",
        "Alexis Ajinca": "Center",
        "Dante Cunningham": "Power forward / small forward",
        "Chris Kaman": "Center",
        "Gerald Henderson": "Point guard",
        "Al-Farouq Aminu": "Small forward / power forward",
        "Ed Davis": "Power forward / center",
        "Damian Lillard": "Point guard",
        "Meyers Leonard": "Center / power forward",
        "Maurice Harkless": "Small forward",
        "Allen Crabbe": "Shooting guard / small forward",
        "CJ McCollum": "Shooting guard / point guard",
        "Mason Plumlee": "Center",
        "Noah Vonleh": "Power forward / center",
        "Tim Frazier": "Point guard",
        "Pat Connaughton": "Shooting guard",
        "Avery Bradley": "Shooting guard / point guard",
        "Isaiah Thomas": "Point guard",
        "Jared Sullinger": "Power forward / center",
        "Jonas Jerebko": "Power forward",
        "Evan Turner": "Small forward / guard",
        "Terry Rozier": "Shooting guard / point guard",
        "James Young": "Small forward / shooting guard",
        "RJ Hunter": "Shooting guard",
        "Kelly Olynyk": "Center / power forward",
        "David Lee": "Power forward / center",
        "Tyler Zeller": "Center",
        "Amir Johnson": "Power forward / center",
        "Jae Crowder": "Power forward / small forward",
        "Iman Shumpert": "Shooting guard / small forward",
        "Nick Collison": "Power forward / center",
        "Steve Novak": "Power forward / small forward",
        "Kevin Durant": "Power forward / small forward",
        "Russell Westbrook": "Point guard",
        "D.J. Augustin": "Point guard",
        "Serge Ibaka": "Center / power forward",
        "Enes Kanter": "Center",
        "Kyle Singler": "Small forward",
        "Dion Waiters": "Shooting guard",
        "Andre Roberson": "Shooting guard / small forward",
        "Steven Adams": "Center",
        "Josh Huestis": "Small forward / power forward",
        "Cameron Payne": "Point guard",
        "Kyrie Irving": "Shooting guard / point guard",
        "Tony Wroten": "Point guard / shooting guard",
        "Cleanthony Early": "Power forward",
        "Andrew Bogut": "Center",
        "Festus Ezeli": "Center",
        "James Michael McAdoo": "Power forward",
        "Klay Thompson": "Shooting guard / small forward",
        "Draymond Green": "Power forward",
        "Jason Thompson": "Center / power forward",
        "Brandon Rush": "Shooting guard / small forward",
        "Marreese Speights": "Center / power forward",
        "Stephen Curry": "Point guard",
        "Ian Clark": "Shooting guard",
        "Shaun Livingston": "Shooting guard",
        "Andre Iguodala": "Small forward",
        "Leandro Barbosa": "Shooting guard / point guard",
        "Luis Montero": "Shooting guard",
        "Devin Booker": "Shooting guard",
        "Brandon Knight": "Point guard / shooting guard",
        "Tyson Chandler": "Center",
        "Bryce Cotton": "Point guard / shooting guard",
        "Sonny Weems": "Small forward / shooting guard",
        "TJ Warren": "Small forward / power forward",
        "Ronnie Price": "Point guard",
        "PJ Tucker": "Power forward / small forward",
        "Archie Goodwin": "Shooting guard",
        "Alex Len": "Center",
        "Jon Leuer": "Power forward / center",
        "Mirza Teletovic": "Power forward",
        "Cory Jefferson": "Power forward",
        "Aaron Gordon": "Power forward / small forward",
        "Dewayne Dedmon": "Center",
        "Elfrid Payton": "Point guard",
        "Victor Oladipo": "Shooting guard",
        "Channing Frye": "Power forward / center",
        "Nikola Vucevic": "Center",
        "Evan Fournier": "Shooting guard / small forward",
        "Tobias Harris": "Power forward / small forward",
        "Shabazz Napier": "Point guard",
        "Jason Smith": "Center / power forward",
        "Mario Hezonja": "Small forward / power forward",
        "Andrew Nicholson": "Power forward",
        "James Johnson": "Power forward / small forward",
        "Luis Scola": "Power forward",
        "Cory Joseph": "Point guard",
        "Kyle Lowry": "Point guard",
        "Bismack Biyombo": "Center",
        "DeMar DeRozan": "Shooting guard / small forward",
        "Jonas Valanciunas": "Center",
        "Bruno Caboclo": "Power forward",
        "Norman Powell": "Shooting guard",
        "Terrence Ross": "Shooting guard / small forward",
        "Patrick Patterson": "Power forward",
        "Delon Wright": "Point guard / shooting guard",
        "Lucas Nogueira": "Center",
        "Nene Hilario": "Center / power forward",
        "Jarell Eddie": "Small forward",
        "Gorgui Dieng": "Center",
        "Zach LaVine": "Shooting guard / small forward",
        "Ricky Rubio": "Point guard",
        "Tayshaun Prince": "Small forward",
        "Nikola Pekovic": "Center",
        "Shabazz Muhammad": "Shooting guard / small forward",
        "Kevin Garnett": "Power forward",
        "Andrew Wiggins": "Small forward",
        "Kevin Martin": "Shooting guard",
        "Andre Miller": "Point guard",
        "Karl-Anthony Towns": "Power forward / center",
        "Adreian Payne": "Center / power forward",
        "Nemanja Bjelica": "Power forward",
        "Carl Landry": "Power forward",
        "Dirk Nowitzki": "Power forward / center",
        "Zaza Pachulia": "Center",
        "Devin Harris": "Point guard",
        "Raymond Felton": "Point guard / shooting guard",
        "Charlie Villanueva": "Power forward",
        "Deron Williams": "Point guard",
        "Jose Juan Barea": "Point guard",
        "JaVale McGee": "Center",
        "Wesley Matthews": "Shooting guard / small forward",
        "Jeremy Evans": "Power forward / center",
        "Chandler Parsons": "Small forward",
        "Dwight Powell": "Center / power forward",
        "Justin Anderson": "Shooting guard / small forward",
        "Jason Terry": "Shooting guard / point guard",
        "Dwight Howard": "Center",
        "Trevor Ariza": "Small forward",
        "Corey Brewer": "Small forward",
        "James Harden": "Point guard / shooting guard",
        "Ty Lawson": "Point guard",
        "Patrick Beverley": "Point guard / shooting guard",
        "Marcus Thornton": "Shooting guard",
        "Terrence Jones": "Power forward",
        "KJ McDaniels": "Small forward",
        "Clint Capela": "Center",
        "Montrezl Harrell": "Center / power forward",
        "Harrison Barnes": "Small forward",
        "Shane Larkin": "Point guard / shooting guard",
        "Joe Johnson": "Shooting guard / small forward",
        "Andrea Bargnani": "Power forward / center",
        "Sergey Karasev": "Small forward / shooting guard",
        "Brook Lopez": "Center",
        "Donald Sloan": "Point guard / shooting guard",
        "Wayne Ellington": "Shooting guard",
        "Markel Brown": "Shooting guard / point guard",
        "Thaddeus Young": "Power forward",
        "Willie Reed": "Center",
        "Thomas Robinson": "Center / power forward",
        "Bojan Bogdanovic": "Small forward",
    }

    return get_basketball_game(
        path_to_game_data,
        path_to_event_label_data,
        path_to_baller2vec_info,
        CLE_POTENTIAL_STARTER_NAMES_2_ENTITY_IDXS,
        PLAYER_NAMES_IN_DATASET_2_POSITIONS,
        event_idxs,
        sampling_rate_Hz,
    )
