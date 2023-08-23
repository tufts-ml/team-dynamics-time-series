# Exported Basketball Data Readme

## Data splits 

The `basketball.data.baller2vec.write_to_disk` module constructs the **2015-2016 “Cleveland Starters” Dataset**, a 
dataset of locations of NBA basketball players on the court during games.

This data was created by applying our preprocessing code which was applied to the [baller2vec representation](https://github.com/airalcorn2/baller2vec) of the [NBA-Player-Movements dataset](https://github.com/linouk23/NBA-Player-Movements).

Basically, we took all the available games where one team was the Cleveland Cavs (CLE; who won the NBA championship that year), and where CLE used one of its 4 most common starting lineups.   For each such game, we extracted the plays that contained the CLE starters.

I have split the 29 such games chronologically into training (`g` games), validation (4 games), and test (5 games) sets, where `g` takes the values of 1,5, and 20 for small, medium, and large training datasets, respectively. Note that the differently sized training sets are nested; in particular, the small, medium, and large training sets consist of the **last** 1, 5, and 20 candidate training games, respectively.  This procedure prevents us from conflating the training set's sample size and its chronological proximity to test set.

### Public data  

For each split, the files provide some ``public data" which can be used by _external_ packages for training and forecasting:

* **Player coordinates on the court** (`player_coords_*.py`):  The arrays have the form (T,J=10,D=2), where T is the number of timesteps (when sampling at 5Hz) across ALL selected plays from ALL selected games, J is the number of players and D is the dimensionality of the court.  The player coordinates are _normalized_ to the unit square; that is, the court is mapped from [0,94]x[0,50] feet to [0,1]x[0,1].   Values slightly less than 0 or slightly greater than 1 are possible if the player steps out of bounds.  Note that we assume that CLE always has their hoop on the left; if not, we rotate the court 180 degrees (following baller2vec).
* **Ball coordinates on the court** (`ball_coords_*.py`):  The arrays have the form (T,D=3), where T is the number of timesteps (when sampling at 5Hz) across ALL selected plays from ALL selected games, and D is the dimensionality of the ball.  The ball coordinates are _semi-normalized_; the (x,y) coordinates on the court are mapped from [0,94]x[0,50] feet to [0,1]x[0,1], but the height (z) coordinate is in raw feet [0,\infty).  As with player coordinates, we assume that CLE always has their hoop on the left; if not, we rotate the (x,y) coordinates 180 degrees.
*  **Example stop indices** (`example_stop_idxs_*.npy`): For each datasplit in {train, val, test}, we provide the indices at which an example has ended.   Each example can be treated as a new iid example of a time series model.  An example onset is given by a wall-clock-time gap that is longer than the nominal sampling rate of 5Hz (beyond a small epsilon).   A switch in examples may be caused by halftime, the start of a new game, or omission of filtered-out plays (where the filtered out plays are those that did not contain the CLE starters or which were not correctly represented in the raw dataset).   We store these example stop indices as a vector of form `example_stop_idxs=[-1, t_1, …, t_E]` where `t_e` is the timestep at which the `e`-th example ended.      So to get the timesteps for the `e`-th example, you can index an array as `[example_end_times[e-1]+1 : example_end_times[e]]`.
*  **Player names by example** (`player_names_by_example_*.npy`): This is a numpy array of shape (E,J=10), whose data type is a string.  E refers to the number of examples, and the J=10 columns provide the ordering of the players in the example. We want, and verify in the code, that the player identities are constant for all timesteps within an example.  Using this object, we can determine which player corresponds to, say, index $j=3$ in the player coordinates object.

For the test set, there is one additional file, `random_context_times.npy`, which lists the randomly selected number of timesteps to use as context when forecasting on each event in the test set.   If the event was not sufficiently long (above a specified threshold), then the element in the array is `np.nan`.   That means to SKIP using that event for forecasting.


### Private data  

Note that the splits also contain "private data" which is meant only for _internal_ (`dynagroup` package) use.  This information is used for diagnostics, animations, etc.  Some of these files are pickle files which therefore require `dynagroup` to be installed and may become unuseable as the package evolves. Examples include:

* **Events** (`events_*.npy`): These contain "events" in the sense of the `NBA-Player-Movements` and `baller2vec` packages. They tend to correspond to different "plays", such as a rebound.  
* **Play stop start indices** (`play_stop_start_idxs_*.npy`)

## Player indexing

The player indexing is as follows.  Indices 0-4 are always reserved for the CLE starters.   The true identities of the CLE starters can vary slightly (at 2/5 positions) from game to game, but take the form

```
CLE_POTENTIAL_STARTER_NAMES_2_ENTITY_IDXS = {
        "LeBron James": 0,  # everything
        "Kevin Love": 1,  # power forward, center
        "Timofey Mozgov": 2,  # center
        "Tristan Thompson": 2,  # center
        "J.R. Smith": 3,  # shooting guard/small forward
        "Kyrie Irving": 4,  # guard
        "Mo Williams": 4,  # guard
        "Matthew Dellavedova": 4,  # guard
    }
```
   
The indices 5-9 are always reserved for opponents (OPP).  Identities of the OPP players obviously varies widely from game to game, but indices 5-6 are reserved for the OPP team’s forwards, 7  for the OPP center, and 8-9 for the OPP guards. 