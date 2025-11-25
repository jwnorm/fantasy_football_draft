### A Pluto.jl notebook ###
# v0.19.40

using Markdown
using InteractiveUtils

# ╔═╡ 983b81da-c743-11f0-1d86-f93134ca51df
using JuMP, HiGHS, CSV, DataFrames, OrderedCollections

# ╔═╡ f4b6f9a1-ba6b-4072-8474-da83c35357f8
md"""
# Fantasy Football Draft
**Jacob Norman**

* Scenario: Base
* Projections: *The Athletic*
* Scoring: PPR
* Draft Position: 6
* Draft Value: ADP
* Tight Ends: Yes
"""

# ╔═╡ db4365ba-9d00-4c52-b1a8-49b7698cd7bf
md"""
## Introduction

This document is a walkthrough of a fantasy football draft optimization model developed as the final project for *OR 708: Integer Programming* at NC State. This is building off a similar model I created for fantasy baseball during my very first semester of my masters program.

The goal of the project is to:
1) Successfully model a fantasy football draft as a **B**inary **I**nteger **P**rogramming (BIP) model;
2) Compare this against actual data through Week 11 of the 2025 NFL season; and
3) Conduct a scenario analysis by changing various aspects of the draft

This report will be concerned with the first two items. Before we begin, we will import the required libraries, including `HiGHS` as the solver.
"""

# ╔═╡ 8f3e1c46-2b62-40d6-8f22-f7ecbb1a9d92
md"""
Next, let's read in our data file that includes the players and their various statistics. The top 208 players drafted during [**N**ational **F**antasy **C**hampionship (NFC)] (https://nfc.shgn.com/adp/football) leagues in the week leading up to the start of the NFL regular season are included, along with their **A**verage **D**raft **P**osition (ADP) information. The preseason projections were obtained from [The Athletic's Jake Ciely](https://www.nytimes.com/athletic/6432965/2025/06/19/fantasy-football-2025-rankings-projections-cheat-sheet/), and are scored using **P**oints **P**er **R**eception (PPR).
"""

# ╔═╡ f5b3c6fa-3fc7-4a73-a0b3-6e77cc0391f8
df = DataFrame(CSV.File("../data/fantasy_football_main_data.csv"))

# ╔═╡ 24248c32-7214-48cc-8606-5235ca0c9682
md"""
## Sets & Parameters

Before we build the BIP model, it is important to define some important parameters that will govern how the model will behave. The most popular fantasy leagues have 12 teams with a roster size of 16 NFL players; this means that the fantasy draft is 16 rounds as well. This is also the case with my real-life fantasy league that I am in this year.

A common strategy in almost all leagues is to draft a defense and kicker in the final two rounds as these are the least important positions. To simplfy the model, we will exclude these positions. This means the draft will actually only be 14 rounds total.

Let's define the index sets below:

$i \in M = \{\text{Ja'Marr Chase}, \text{Bijan Robinson}, \dots, \text{Joe Flacco}\}$
$j \in N = \{1, 2, \dots, 14\}$

"""

# ╔═╡ 823fe90d-c475-4878-b40d-3bcd8d820f2d
begin
	M = length(df[:, :player_name])		# number of NFL players
	N = 14 								# number of rounds in draft
end

# ╔═╡ a3437289-87cc-4750-a4f0-2df5674b3e29
md"""
As mentioned above, there are 12 teams total in the league. This means there will be 168 picks in total, which is less the the player pool of 208. We also need to define when we will be picking in the first round. To start, let's pick right in the middle in the sixth position.
"""

# ╔═╡ e50e8537-775e-4846-ba3b-6488806a5ded
begin
	teams = 12 			 # number of teams in league
	start_position = 6   # starting draft position in round one
end

# ╔═╡ 201d4ec2-9c4b-4e56-89e7-ec98d19dfcad
md"""
Each team has something called a bye week, which is essentially a week of rest from playing games. This is a crucial thing to consider in a fantasy league because too many players on bye at one time can lead to an easy loss. There are 18 weeks in an NFL season, but not all of them are bye weeks; let's create a vector defining the different week numbers that correspond to bye weeks.
"""

# ╔═╡ e1fe18c7-852f-45d1-bfc3-edf3d0a86046
bye_weeks = ["5", "6", "7", "8", "9", "10", "11", "12", "14"]

# ╔═╡ acffe2ea-e76e-4385-8084-57eb22847567
md"""
Finally, we need to consider roster construction. Excluding defense and kicker, a standard roster is composed of the follow positions:

* **Q**uarter**B**ack (QB): 1
* **R**unning**B**ack (RB): 2
* **W**ide **R**eceiver (WR): 2
* **T**ight **E**nd (TE): 1
* Flex: 2
* Bench: 6

The flex position includes RB-, WR-, or TE-eligible players and the bench is reserved for players who do not start during the week. While they do not directly contribute to the weekly score, they are helpful when managing bye weeks and they could transition to starters themselves if they play well enough throughout the year.
"""

# ╔═╡ b7271ab2-1cdd-4819-a6af-09addef53370
position_requirements = OrderedDict(
	"qb" => 1, 
	"rb" => 2, 
	"wr" => 2, 
	"te" => 1, 
	"flex" => 2
)

# ╔═╡ 8460c50c-9e78-48d6-926c-c5def0707f0e
md"""
## Decision Variables & Objective Function

Now we can get into building the BIP. There will be almost 3,000 binary decision variables, which correspond to the player and the round they were selected in:

$x_{ij} \in \{0, 1\}, \ for\ i \in M, \ j \in N$
"""

# ╔═╡ b6f4bd46-fcd7-4818-86d1-539092c4dd58
begin
	# initialize BIP model
	model = Model(HiGHS.Optimizer)

	# define decision variables
	@variable(model,
		  x[df[:, :player_name], 1:N], 
		  Bin)
end

# ╔═╡ c8b692cb-3ec7-42da-80d3-7459df1ae2c3
md"""
We are interested in maximizing the total number of projected fantasy points accumulated by the players, denoted by $c_j$, that we select in the draft:

$\text{max} \ z = \sum_{(i,j)}{c_{j} x_{ij}}$
"""

# ╔═╡ aa368c66-a628-4266-9a9b-787139836bea
@objective(model, 
		   Max, 
		   sum(x[df[i, :player_name], j] * df[i, :athletic_ppr_projected_points] 
		   for i=1:M, j=1:N))

# ╔═╡ cf3aa873-a855-45ce-9e67-6aea9e161268
md"""
## Constraints

This problem is conceptually very similar to the assignment problem, with some additional constraints. These include:

* One player *must* be selected each round
* A player can be selected at most once
* Draft position enforcement, broken down into two groups of constraints
* Positional requirements must be met
* A maximum of two QBs can be selected
* There can be at most three players selected that have the same bye week

These will defined in order below, beginning with the requirement that exactly one player is selected each round:

$\sum_{i}{x_{ij}} = 1,\ \text{for}\ j=1, \dots, 14$
"""

# ╔═╡ 1566b406-89b8-4d63-9cf6-efa4f66b8574
# one player must be selected per round
@constraint(model, 
			round_max[j ∈ 1:N], 
			sum(x[:, j]) == 1)

# ╔═╡ c354d05a-66b0-4b32-9414-140084ab3057
md"""
Next, we need to make sure that a player is not selected more than once:

$\sum_{j=1}^{14}{x_{ij}} ≤ 1,\ \text{for}\ i \in M$
"""

# ╔═╡ 3e9ac665-89fd-4ab0-a789-6a4e3700e722
# a player can be selected at most once
@constraint(model, 
			player_max[i ∈ df[:, :player_name]], 
			sum(x[i, :]) <= 1)

# ╔═╡ 80a752e4-a209-44fd-a886-a727e215d340
md"""
To enforce the position at which players are able to be selected, we require two groups of constraints; this is due to the snake draft format. Defining ADP as $adp_i$, we have the following constraints for odd rounds:

$\sum_{i}{adp_{i}x_{ij}} ≥ 12(j-1) + 6,\ \text{for}\ j=1,3,5,...,13$

Similarly, for even rounds we have:

$\sum_{i}{adp_{i}x_{ij}} ≥ 12(j-1) + 7,\ \text{for}\ j=2,4,6,...,14$
"""

# ╔═╡ f58d5443-fa5e-4dcb-8ce4-e5991c135bd7
# draft position enforcement: odd rounds
@constraint(model, 
			adp_odd_max[j ∈ 1:2:N], 
			sum(df[i, :adp] * x[df[i, :player_name], j] 
			for i=1:M) >= teams * (j - 1) + start_position)

# ╔═╡ bfddf2d6-6705-43f5-a159-80f1ba09444e
# draft position enforcement: even rounds
@constraint(model, 
			adp_even_max[j ∈ 2:2:N], 
			sum(df[i, :adp] * x[df[i, :player_name], j] 
			for i=1:M) >= teams * (j - 1) + start_position + 1)

# ╔═╡ c90b82df-a153-4b4e-8aa0-22fb11d57626
md"""
Now, we need to ensure that each position has at least the minimum number of eligible players. Let us define $p_{ik}$ as $1$ if player $i$ is eligible to play position $k$, and $0$ otherwise. We will also define $pos_k$ as the minimum required players for position $k$:

$\sum_{i}\sum_{j=1}^{14}{p_{ik}x_{ij}} ≥ pos_{p},\ \text{for}\ k \in \{\text{QB}, \text{RB}, \text{WR}, \text{TE}, \text{Flex}\}$
"""

# ╔═╡ 391f1ec6-d519-4847-9a75-a6e51a416986
# required number of each position must be met
@constraint(model, 
			pos_req[k ∈ keys(position_requirements)], 
			sum(df[i, "is_"*k*"_eligible"] * x[df[i, :player_name], j] 
			for i=1:M, j=1:N) >= position_requirements[k])

# ╔═╡ 67060b03-684b-4191-9119-65d6994faf64
md"""
We now need to limit the number of quarterbacks selected to two. If we do not, it is likely the model will fill the entire bench with quarterbacks since they typically score the most points among the different position groups. It is standard practice to only select two quarterbacks in a standard draft anyway.

$\sum_{i}\sum_{j=1}^{14}{p_{i,\text{QB}}x_{ij}} \le 2$
"""

# ╔═╡ b1d278e8-6af7-4b8f-8fb8-061b1bc78d0a
# at most two qbs can be selected
@constraint(model, 
			sum(df[i, :is_qb_eligible] * x[df[i, :player_name], j] 
			for i=1:M, j=1:N) <= 2)

# ╔═╡ 9ac8c40f-0625-4803-a5aa-8322d7051afe
md"""
Lastly, we need to ensure that no more than three players share the same bye week. We will define $b_{ir}$ as $1$ if player $i$ has a bye during week $r$, and $0$ otherwise.

$\sum_{i}\sum_{j=1}^{14}{b_{ir}x_{ij}} \le 3,\ \text{for}\ r \in \{5, 6, 7, 8, 9, 10, 11, 12, 14\}$
"""

# ╔═╡ 488b3b78-be30-4fa0-a3dd-08f3029f4a98
# at most 3 players can have the same bye week
@constraint(model, 
			bye[b ∈ bye_weeks], 
			sum(df[i, "is_bye_week"*b] * x[df[i, :player_name], j] 
			for i=1:M, j=1:N) <= 3)

# ╔═╡ f17de48c-8ac0-4637-94a4-5c9069525890
md"""
## Solution

Now that the entire model is defined, we can actually use `HiGHS` to solve it.
"""

# ╔═╡ ae036656-b16c-488e-b4d0-ebbad54fe5f0
begin
	optimize!(model)
	#solution_summary(model)
end

# ╔═╡ 9bde5945-9777-4454-aceb-ddd645acc404
md"""
We see that the optimal value to our fantasy draft problem is $(Int(round(objective_value(model); digits=0))). Let's see the breakdown of the optimal team and when each player was selected in the draft.
"""

# ╔═╡ 256c5aeb-9bbf-416b-bce3-43bf7cdd08fb
begin
	# initialized all players with 0
	drafted_players = OrderedDict(i => 0 for i ∈ df[:, :player_name])

	# assign round to players that were drafted
	for j in 1:N, i ∈ df[:, :player_name]
	    if round(value(x[i, j])) == 1
			drafted_players[i] = j
		end
	end

	# add to df
	df[:, :round] = [drafted_players[i] for i ∈ df[:, :player_name]]

	# create new df
	drafted_players_df = filter(:round => >=(1), df)
	sort!(drafted_players_df, order(:round))

	# display roster
	cols = [:round, :player_name, :bye_week, :adp, :position, :athletic_ppr_projected_points]
	drafted_players_df[:, cols]
end

# ╔═╡ dbc5d90d-d6d1-4d72-95a3-1707a4fc3697
md"""
Let's confirm that are constraints are being met:

* No players is selected more than once
* No rounds have more than one selection
* There are 2 QBs, 4 RBs, 6 WRs, and 2 TEs
* There are 3 players with bye week 8; all others are 2 or under
* The draft position constraints are correctly holding

Looks like this is indeed a feasible solution! Now we can turn our attention to analysis of the drafted roster. The draft strategy can best be described as *heroRB*, meaning that the first several selections are all top-end running backs. This includes Christian McCaffrey, who has been the overall top player in fantasy several times, and Bucky Irving, a talented second year player hoping to have a jump in performance. This is notable because PPR leagues generally see a higher value placed on wide receivers since they catch the ball more often.

Another interesting note is that both quarterbacks were taken towards the very end of the draft. This is a very common draft strategy that we did not specifically tell the model to follow, it chose to on its own.
"""

# ╔═╡ 26325d60-d99f-4887-a09b-0945940a8bf4
md"""
## Comparison to Actual Results


Now that we have the optimal roster based on preseason projections, let's run the same model, but this time use actual fantasy points through Week 11 of the NFL season. This data was sourced from [**P**ro **F**ootball **F**ocus (PFF)](https://www.pff.com/fantasy/stats).

To make this a little easier, let's put all of the relevant code into a function that we can run with a single click:

"""

# ╔═╡ 3c3d9a5e-9bb1-46bb-bbb7-ca02f9e993ca
"""
This function solves the binary integer programming model for the fantasy football draft using actual PPR fantasy points scored through Week 11 of the 2025 NFL season.

Arguments

- `position_requirements::OrderedDict{String, Int64}`: A dictionary of the roster positions and the minimum number of eligible players for each one
- `start_position::Int64`, optional: Starting draft position in round one; default is `6`
- `verbose::Bool`, optional: Print total projected points to screen; default is `true`


Returns   

- `x::Any`: Optimal x values from model results
"""
function solve_true_model(position_requirements::OrderedDict{String, Int64}; 	
							start_position::Int64=6, verbose::Bool=true)

	# read in projection df
	df = DataFrame(CSV.File("../data/fantasy_football_main_data.csv"))

	# initialize model
	model = Model(HiGHS.Optimizer)
	set_silent(model)

	# sets
	M = length(df[:, :player_name])		# number of NFL players
	N = 14 								# number of rounds in draft

	# parameters
	teams = 12
	bye_weeks = ["5", "6", "7", "8", "9", "10", "11", "12", "14"]

	# decision variables
	@variable(model, x[df[:, :player_name], 1:N], Bin)

	# objective function
	@objective(model, 
		   	   Max, 
		       sum(x[df[i, :player_name], j] * df[i, :week11_actual_ppr_points] 
		       for i=1:M, j=1:N))

	# constraints

	# one player must be selected per round
	@constraint(model, 
				round_max[j ∈ 1:N], 
				sum(x[:, j]) == 1)

	
	# a player can be selected at most once
	@constraint(model, 
				player_max[i ∈ df[:, :player_name]], 
				sum(x[i, :]) <= 1)

	# draft position enforcement: odd rounds
	@constraint(model, 
				adp_odd_max[j ∈ 1:2:N], 
				sum(df[i, :adp] * x[df[i, :player_name], j] 
				for i=1:M) >= teams * (j - 1) + start_position)

	# draft position enforcement: even rounds
	@constraint(model, 
				adp_even_max[j ∈ 2:2:N], 
				sum(df[i, :adp] * x[df[i, :player_name], j] 
				for i=1:M) >= teams * (j - 1) + start_position + 1)

	# required number of each position must be met
	@constraint(model, 
				pos_req[k ∈ keys(position_requirements)], 
				sum(df[i, "is_"*k*"_eligible"] * x[df[i, :player_name], j] 
				for i=1:M, j=1:N) >= position_requirements[k])

	# at most two qbs can be selected
	@constraint(model, 
				sum(df[i, :is_qb_eligible] * x[df[i, :player_name], j] 
				for i=1:M, j=1:N) <= 2)

	# at most 3 players can have the same bye week
	@constraint(model, 
				bye[b ∈ bye_weeks], 
				sum(df[i, "is_bye_week"*b] * x[df[i, :player_name], j] 
				for i=1:M, j=1:N) <= 3)
	
	# solve model
	optimize!(model)

	if verbose
		# print optimal value
		optimal_value = Int(round(objective_value(model); digits=0))
		print("Total Projected Points:\t", optimal_value)
	end
	
	# return optimal x values
	return x
end

# ╔═╡ 8fe8336e-2a6f-4376-9881-34293303e97a
x_true = solve_true_model(position_requirements)

# ╔═╡ 7c074634-043a-4721-81f1-352d45c67ea2
md"""
Now that we have our new model results, `x_true`, let's create a function that displays them in the same format that we defined above.
"""

# ╔═╡ d6efb1a4-fd99-40e2-a4d2-11d006969750
"""
This displays the output from a `JuMP` model for the fantasy football draft binary integer programming model as a `DataFrame`.

Arguments

- `x::Any`: Optimal x values from model results

Returns   

- `clean_drafted_players::DataFrame`: Output of drafted players along with some descriptive statistics and their fantasy point totals.
"""
function display_drafted_players(x)

	# initialize df
	roster = deepcopy(df)

	# initialized all players with 0
	drafted_players = OrderedDict(i => 0 for i ∈ roster[:, :player_name])

	# assign round to players that were drafted
	for j in 1:N, i ∈ roster[:, :player_name]
	    if round(value(x[i, j])) == 1
			drafted_players[i] = j
		end
	end

	# add to df
	roster[:, :round] = [drafted_players[i] for i ∈ roster[:, :player_name]]

	# create new df
	drafted_players_df = filter(:round => >=(1), roster)
	sort!(drafted_players_df, order(:round))

	# display roster
	cols = [:round, :player_name, :bye_week, :adp, :position, :week11_actual_ppr_points]
	clean_drafted_players = drafted_players_df[:, cols]
	
	return clean_drafted_players
	
end

# ╔═╡ c567c1f2-1049-4285-a5f8-bc0d99b390da
begin
	true_drafted_players_df = display_drafted_players(x_true)
	true_drafted_players_df
end

# ╔═╡ 4e9c8d2d-a120-4982-858a-f6cc1743447a
md"""
If you have been following the 2025 NFL season, this list comes as no surpise. Jonathan Taylor has been the overall best player in all of fantasy, Emeka Egbuka is a dynamic rookie wide reciever that is a great value at his draft position, and Drake Maye is in the midst of an MVP season as the quarterback for the New England Patriots. The only player that overlaps from our preseason model is Christian McCaffrey, who was selected in the first round in both cases. This illustrates the variability that is observed in the NFL from year to year.

The actual draft strategy did not change with the benefit for foresight; the model still chose to heavily favor running backs early in the draft. This further suggests that the *heroRB* strategy makes sense to implement in PPR leagues.

Using the Week 11 actual points, let's see how our **base** drafted team compares to **true** roster, in terms of total points.
"""

# ╔═╡ b9824aaa-0943-407d-9271-9792b263c340
begin
	base_total = sum(drafted_players_df[:, :week11_actual_ppr_points])
	true_total = sum(true_drafted_players_df[:, :week11_actual_ppr_points])

	base_correct_pct = base_total / true_total
end;

# ╔═╡ 67e5cc65-2a07-4639-89b1-ab88b00658f6
md"""
The base model has attained $(round(base_correct_pct * 100; digits=1))% of the total possible points through Week 11. This further asserts the variation and randomness inherent in an NFL season. After all, if it was this simple to select the perfect roster, fantasy football would be quite boring.
"""

# ╔═╡ 00000000-0000-0000-0000-000000000001
PLUTO_PROJECT_TOML_CONTENTS = """
[deps]
CSV = "336ed68f-0bac-5ca0-87d4-7b16caf5d00b"
DataFrames = "a93c6f00-e57d-5684-b7b6-d8193f3e46c0"
HiGHS = "87dc4568-4c63-4d18-b0c0-bb2238e4078b"
JuMP = "4076af6c-e467-56ae-b986-b466b2749572"
OrderedCollections = "bac558e1-5e72-5ebc-8fee-abe8a469f55d"

[compat]
CSV = "~0.10.15"
DataFrames = "~1.7.0"
HiGHS = "~1.14.0"
JuMP = "~1.24.0"
OrderedCollections = "~1.8.0"
"""

# ╔═╡ 00000000-0000-0000-0000-000000000002
PLUTO_MANIFEST_TOML_CONTENTS = """
# This file is machine-generated - editing it directly is not advised

julia_version = "1.10.2"
manifest_format = "2.0"
project_hash = "95b6d7c5ac21d9ed3e8a14d43b471a734c180dd8"

[[deps.Artifacts]]
uuid = "56f22d72-fd6d-98f1-02f0-08ddc0907c33"

[[deps.Base64]]
uuid = "2a0f44e3-6c83-55bd-87e4-b1978d98bd5f"

[[deps.BenchmarkTools]]
deps = ["Compat", "JSON", "Logging", "Printf", "Profile", "Statistics", "UUIDs"]
git-tree-sha1 = "e38fbc49a620f5d0b660d7f543db1009fe0f8336"
uuid = "6e4b80f9-dd63-53aa-95a3-0cdb28fa8baf"
version = "1.6.0"

[[deps.Bzip2_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "1b96ea4a01afe0ea4090c5c8039690672dd13f2e"
uuid = "6e34b625-4abd-537c-b88f-471c36dfa7a0"
version = "1.0.9+0"

[[deps.CSV]]
deps = ["CodecZlib", "Dates", "FilePathsBase", "InlineStrings", "Mmap", "Parsers", "PooledArrays", "PrecompileTools", "SentinelArrays", "Tables", "Unicode", "WeakRefStrings", "WorkerUtilities"]
git-tree-sha1 = "deddd8725e5e1cc49ee205a1964256043720a6c3"
uuid = "336ed68f-0bac-5ca0-87d4-7b16caf5d00b"
version = "0.10.15"

[[deps.CodecBzip2]]
deps = ["Bzip2_jll", "TranscodingStreams"]
git-tree-sha1 = "84990fa864b7f2b4901901ca12736e45ee79068c"
uuid = "523fee87-0ab8-5b00-afb7-3ecf72e48cfd"
version = "0.8.5"

[[deps.CodecZlib]]
deps = ["TranscodingStreams", "Zlib_jll"]
git-tree-sha1 = "962834c22b66e32aa10f7611c08c8ca4e20749a9"
uuid = "944b1d66-785c-5afd-91f1-9de20f533193"
version = "0.7.8"

[[deps.CommonSubexpressions]]
deps = ["MacroTools"]
git-tree-sha1 = "cda2cfaebb4be89c9084adaca7dd7333369715c5"
uuid = "bbf7d656-a473-5ed7-a52c-81e309532950"
version = "0.3.1"

[[deps.Compat]]
deps = ["TOML", "UUIDs"]
git-tree-sha1 = "8ae8d32e09f0dcf42a36b90d4e17f5dd2e4c4215"
uuid = "34da2185-b29b-5c13-b0c7-acf172513d20"
version = "4.16.0"
weakdeps = ["Dates", "LinearAlgebra"]

    [deps.Compat.extensions]
    CompatLinearAlgebraExt = "LinearAlgebra"

[[deps.CompilerSupportLibraries_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "e66e0078-7015-5450-92f7-15fbd957f2ae"
version = "1.1.0+0"

[[deps.Crayons]]
git-tree-sha1 = "249fe38abf76d48563e2f4556bebd215aa317e15"
uuid = "a8cc5b0e-0ffa-5ad4-8c14-923d3ee1735f"
version = "4.1.1"

[[deps.DataAPI]]
git-tree-sha1 = "abe83f3a2f1b857aac70ef8b269080af17764bbe"
uuid = "9a962f9c-6df0-11e9-0e5d-c546b8b5ee8a"
version = "1.16.0"

[[deps.DataFrames]]
deps = ["Compat", "DataAPI", "DataStructures", "Future", "InlineStrings", "InvertedIndices", "IteratorInterfaceExtensions", "LinearAlgebra", "Markdown", "Missings", "PooledArrays", "PrecompileTools", "PrettyTables", "Printf", "Random", "Reexport", "SentinelArrays", "SortingAlgorithms", "Statistics", "TableTraits", "Tables", "Unicode"]
git-tree-sha1 = "fb61b4812c49343d7ef0b533ba982c46021938a6"
uuid = "a93c6f00-e57d-5684-b7b6-d8193f3e46c0"
version = "1.7.0"

[[deps.DataStructures]]
deps = ["Compat", "InteractiveUtils", "OrderedCollections"]
git-tree-sha1 = "1d0a14036acb104d9e89698bd408f63ab58cdc82"
uuid = "864edb3b-99cc-5e75-8d2d-829cb0a9cfe8"
version = "0.18.20"

[[deps.DataValueInterfaces]]
git-tree-sha1 = "bfc1187b79289637fa0ef6d4436ebdfe6905cbd6"
uuid = "e2d170a0-9d28-54be-80f0-106bbe20a464"
version = "1.0.0"

[[deps.Dates]]
deps = ["Printf"]
uuid = "ade2ca70-3891-5945-98fb-dc099432e06a"

[[deps.DiffResults]]
deps = ["StaticArraysCore"]
git-tree-sha1 = "782dd5f4561f5d267313f23853baaaa4c52ea621"
uuid = "163ba53b-c6d8-5494-b064-1a9d43ac40c5"
version = "1.1.0"

[[deps.DiffRules]]
deps = ["IrrationalConstants", "LogExpFunctions", "NaNMath", "Random", "SpecialFunctions"]
git-tree-sha1 = "23163d55f885173722d1e4cf0f6110cdbaf7e272"
uuid = "b552c78f-8df3-52c6-915a-8e097449b14b"
version = "1.15.1"

[[deps.DocStringExtensions]]
deps = ["LibGit2"]
git-tree-sha1 = "2fb1e02f2b635d0845df5d7c167fec4dd739b00d"
uuid = "ffbed154-4ef7-542d-bbb7-c09d3a79fcae"
version = "0.9.3"

[[deps.FilePathsBase]]
deps = ["Compat", "Dates"]
git-tree-sha1 = "3bab2c5aa25e7840a4b065805c0cdfc01f3068d2"
uuid = "48062228-2e41-5def-b9a4-89aafe57970f"
version = "0.9.24"
weakdeps = ["Mmap", "Test"]

    [deps.FilePathsBase.extensions]
    FilePathsBaseMmapExt = "Mmap"
    FilePathsBaseTestExt = "Test"

[[deps.ForwardDiff]]
deps = ["CommonSubexpressions", "DiffResults", "DiffRules", "LinearAlgebra", "LogExpFunctions", "NaNMath", "Preferences", "Printf", "Random", "SpecialFunctions"]
git-tree-sha1 = "a2df1b776752e3f344e5116c06d75a10436ab853"
uuid = "f6369f11-7733-5829-9624-2563aa707210"
version = "0.10.38"

    [deps.ForwardDiff.extensions]
    ForwardDiffStaticArraysExt = "StaticArrays"

    [deps.ForwardDiff.weakdeps]
    StaticArrays = "90137ffa-7385-5640-81b9-e52037218182"

[[deps.Future]]
deps = ["Random"]
uuid = "9fa8497b-333b-5362-9e8d-4d0656e87820"

[[deps.HiGHS]]
deps = ["HiGHS_jll", "MathOptInterface", "PrecompileTools", "SparseArrays"]
git-tree-sha1 = "0938730463f925e04a52c82335b78ff1209b29e8"
uuid = "87dc4568-4c63-4d18-b0c0-bb2238e4078b"
version = "1.14.0"

[[deps.HiGHS_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "JLLWrappers", "Libdl", "Zlib_jll"]
git-tree-sha1 = "26694f04567e584b054b9f33a810cec52adafa38"
uuid = "8fd58aa0-07eb-5a78-9b36-339c94fd15ea"
version = "1.9.0+0"

[[deps.InlineStrings]]
git-tree-sha1 = "6a9fde685a7ac1eb3495f8e812c5a7c3711c2d5e"
uuid = "842dd82b-1e85-43dc-bf29-5d0ee9dffc48"
version = "1.4.3"

    [deps.InlineStrings.extensions]
    ArrowTypesExt = "ArrowTypes"
    ParsersExt = "Parsers"

    [deps.InlineStrings.weakdeps]
    ArrowTypes = "31f734f8-188a-4ce0-8406-c8a06bd891cd"
    Parsers = "69de0a69-1ddd-5017-9359-2bf0b02dc9f0"

[[deps.InteractiveUtils]]
deps = ["Markdown"]
uuid = "b77e0a4c-d291-57a0-90e8-8db25a27a240"

[[deps.InvertedIndices]]
git-tree-sha1 = "6da3c4316095de0f5ee2ebd875df8721e7e0bdbe"
uuid = "41ab1584-1d38-5bbf-9106-f11c6c58b48f"
version = "1.3.1"

[[deps.IrrationalConstants]]
git-tree-sha1 = "e2222959fbc6c19554dc15174c81bf7bf3aa691c"
uuid = "92d709cd-6900-40b7-9082-c6be49f344b6"
version = "0.2.4"

[[deps.IteratorInterfaceExtensions]]
git-tree-sha1 = "a3f24677c21f5bbe9d2a714f95dcd58337fb2856"
uuid = "82899510-4779-5014-852e-03e436cf321d"
version = "1.0.0"

[[deps.JLLWrappers]]
deps = ["Artifacts", "Preferences"]
git-tree-sha1 = "a007feb38b422fbdab534406aeca1b86823cb4d6"
uuid = "692b3bcd-3c85-4b1f-b108-f13ce0eb3210"
version = "1.7.0"

[[deps.JSON]]
deps = ["Dates", "Mmap", "Parsers", "Unicode"]
git-tree-sha1 = "31e996f0a15c7b280ba9f76636b3ff9e2ae58c9a"
uuid = "682c06a0-de6a-54ab-a142-c8b1cf79cde6"
version = "0.21.4"

[[deps.JSON3]]
deps = ["Dates", "Mmap", "Parsers", "PrecompileTools", "StructTypes", "UUIDs"]
git-tree-sha1 = "1d322381ef7b087548321d3f878cb4c9bd8f8f9b"
uuid = "0f8b85d8-7281-11e9-16c2-39a750bddbf1"
version = "1.14.1"

    [deps.JSON3.extensions]
    JSON3ArrowExt = ["ArrowTypes"]

    [deps.JSON3.weakdeps]
    ArrowTypes = "31f734f8-188a-4ce0-8406-c8a06bd891cd"

[[deps.JuMP]]
deps = ["LinearAlgebra", "MacroTools", "MathOptInterface", "MutableArithmetics", "OrderedCollections", "PrecompileTools", "Printf", "SparseArrays"]
git-tree-sha1 = "cf832644f225dbe721bb9b97bf432007765fc695"
uuid = "4076af6c-e467-56ae-b986-b466b2749572"
version = "1.24.0"

    [deps.JuMP.extensions]
    JuMPDimensionalDataExt = "DimensionalData"

    [deps.JuMP.weakdeps]
    DimensionalData = "0703355e-b756-11e9-17c0-8b28908087d0"

[[deps.LaTeXStrings]]
git-tree-sha1 = "dda21b8cbd6a6c40d9d02a73230f9d70fed6918c"
uuid = "b964fa9f-0449-5b57-a5c2-d3ea65f4040f"
version = "1.4.0"

[[deps.LibGit2]]
deps = ["Base64", "LibGit2_jll", "NetworkOptions", "Printf", "SHA"]
uuid = "76f85450-5226-5b5a-8eaa-529ad045b433"

[[deps.LibGit2_jll]]
deps = ["Artifacts", "LibSSH2_jll", "Libdl", "MbedTLS_jll"]
uuid = "e37daf67-58a4-590a-8e99-b0245dd2ffc5"
version = "1.6.4+0"

[[deps.LibSSH2_jll]]
deps = ["Artifacts", "Libdl", "MbedTLS_jll"]
uuid = "29816b5a-b9ab-546f-933c-edad1886dfa8"
version = "1.11.0+1"

[[deps.Libdl]]
uuid = "8f399da3-3557-5675-b5ff-fb832c97cbdb"

[[deps.LinearAlgebra]]
deps = ["Libdl", "OpenBLAS_jll", "libblastrampoline_jll"]
uuid = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"

[[deps.LogExpFunctions]]
deps = ["DocStringExtensions", "IrrationalConstants", "LinearAlgebra"]
git-tree-sha1 = "13ca9e2586b89836fd20cccf56e57e2b9ae7f38f"
uuid = "2ab3a3ac-af41-5b50-aa03-7779005ae688"
version = "0.3.29"

    [deps.LogExpFunctions.extensions]
    LogExpFunctionsChainRulesCoreExt = "ChainRulesCore"
    LogExpFunctionsChangesOfVariablesExt = "ChangesOfVariables"
    LogExpFunctionsInverseFunctionsExt = "InverseFunctions"

    [deps.LogExpFunctions.weakdeps]
    ChainRulesCore = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
    ChangesOfVariables = "9e997f8a-9a97-42d5-a9f1-ce6bfc15e2c0"
    InverseFunctions = "3587e190-3f89-42d0-90ee-14403ec27112"

[[deps.Logging]]
uuid = "56ddb016-857b-54e1-b83d-db4d58db5568"

[[deps.MacroTools]]
git-tree-sha1 = "72aebe0b5051e5143a079a4685a46da330a40472"
uuid = "1914dd2f-81c6-5fcd-8719-6d5c9610ff09"
version = "0.5.15"

[[deps.Markdown]]
deps = ["Base64"]
uuid = "d6f4376e-aef5-505a-96c1-9c027394607a"

[[deps.MathOptInterface]]
deps = ["BenchmarkTools", "CodecBzip2", "CodecZlib", "DataStructures", "ForwardDiff", "JSON3", "LinearAlgebra", "MutableArithmetics", "NaNMath", "OrderedCollections", "PrecompileTools", "Printf", "SparseArrays", "SpecialFunctions", "Test"]
git-tree-sha1 = "b691a4b4c8ef7a4fba051d546040bfd2ae6f0719"
uuid = "b8f27783-ece8-5eb3-8dc8-9495eed66fee"
version = "1.37.2"

[[deps.MbedTLS_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "c8ffd9c3-330d-5841-b78e-0817d7145fa1"
version = "2.28.2+1"

[[deps.Missings]]
deps = ["DataAPI"]
git-tree-sha1 = "ec4f7fbeab05d7747bdf98eb74d130a2a2ed298d"
uuid = "e1d29d7a-bbdc-5cf2-9ac0-f12de2c33e28"
version = "1.2.0"

[[deps.Mmap]]
uuid = "a63ad114-7e13-5084-954f-fe012c677804"

[[deps.MutableArithmetics]]
deps = ["LinearAlgebra", "SparseArrays", "Test"]
git-tree-sha1 = "491bdcdc943fcbc4c005900d7463c9f216aabf4c"
uuid = "d8a4904e-b15c-11e9-3269-09a3773c0cb0"
version = "1.6.4"

[[deps.NaNMath]]
deps = ["OpenLibm_jll"]
git-tree-sha1 = "cc0a5deefdb12ab3a096f00a6d42133af4560d71"
uuid = "77ba4419-2d1f-58cd-9bb1-8ffee604a2e3"
version = "1.1.2"

[[deps.NetworkOptions]]
uuid = "ca575930-c2e3-43a9-ace4-1e988b2c1908"
version = "1.2.0"

[[deps.OpenBLAS_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "Libdl"]
uuid = "4536629a-c528-5b80-bd46-f80d51c5b363"
version = "0.3.23+4"

[[deps.OpenLibm_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "05823500-19ac-5b8b-9628-191a04bc5112"
version = "0.8.1+2"

[[deps.OpenSpecFun_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "JLLWrappers", "Libdl"]
git-tree-sha1 = "1346c9208249809840c91b26703912dff463d335"
uuid = "efe28fd5-8261-553b-a9e1-b2916fc3738e"
version = "0.5.6+0"

[[deps.OrderedCollections]]
git-tree-sha1 = "cc4054e898b852042d7b503313f7ad03de99c3dd"
uuid = "bac558e1-5e72-5ebc-8fee-abe8a469f55d"
version = "1.8.0"

[[deps.Parsers]]
deps = ["Dates", "PrecompileTools", "UUIDs"]
git-tree-sha1 = "8489905bcdbcfac64d1daa51ca07c0d8f0283821"
uuid = "69de0a69-1ddd-5017-9359-2bf0b02dc9f0"
version = "2.8.1"

[[deps.PooledArrays]]
deps = ["DataAPI", "Future"]
git-tree-sha1 = "36d8b4b899628fb92c2749eb488d884a926614d3"
uuid = "2dfb63ee-cc39-5dd5-95bd-886bf059d720"
version = "1.4.3"

[[deps.PrecompileTools]]
deps = ["Preferences"]
git-tree-sha1 = "5aa36f7049a63a1528fe8f7c3f2113413ffd4e1f"
uuid = "aea7be01-6a6a-4083-8856-8a6e6704d82a"
version = "1.2.1"

[[deps.Preferences]]
deps = ["TOML"]
git-tree-sha1 = "9306f6085165d270f7e3db02af26a400d580f5c6"
uuid = "21216c6a-2e73-6563-6e65-726566657250"
version = "1.4.3"

[[deps.PrettyTables]]
deps = ["Crayons", "LaTeXStrings", "Markdown", "PrecompileTools", "Printf", "Reexport", "StringManipulation", "Tables"]
git-tree-sha1 = "1101cd475833706e4d0e7b122218257178f48f34"
uuid = "08abe8d2-0d0c-5749-adfa-8a2ac140af0d"
version = "2.4.0"

[[deps.Printf]]
deps = ["Unicode"]
uuid = "de0858da-6303-5e67-8744-51eddeeeb8d7"

[[deps.Profile]]
deps = ["Printf"]
uuid = "9abbd945-dff8-562f-b5e8-e1ebf5ef1b79"

[[deps.Random]]
deps = ["SHA"]
uuid = "9a3f8284-a2c9-5f02-9a11-845980a1fd5c"

[[deps.Reexport]]
git-tree-sha1 = "45e428421666073eab6f2da5c9d310d99bb12f9b"
uuid = "189a3867-3050-52da-a836-e630ba90ab69"
version = "1.2.2"

[[deps.SHA]]
uuid = "ea8e919c-243c-51af-8825-aaa63cd721ce"
version = "0.7.0"

[[deps.SentinelArrays]]
deps = ["Dates", "Random"]
git-tree-sha1 = "712fb0231ee6f9120e005ccd56297abbc053e7e0"
uuid = "91c51154-3ec4-41a3-a24f-3f23e20d615c"
version = "1.4.8"

[[deps.Serialization]]
uuid = "9e88b42a-f829-5b0c-bbe9-9e923198166b"

[[deps.SortingAlgorithms]]
deps = ["DataStructures"]
git-tree-sha1 = "66e0a8e672a0bdfca2c3f5937efb8538b9ddc085"
uuid = "a2af1166-a08f-5f64-846c-94a0d3cef48c"
version = "1.2.1"

[[deps.SparseArrays]]
deps = ["Libdl", "LinearAlgebra", "Random", "Serialization", "SuiteSparse_jll"]
uuid = "2f01184e-e22b-5df5-ae63-d93ebab69eaf"
version = "1.10.0"

[[deps.SpecialFunctions]]
deps = ["IrrationalConstants", "LogExpFunctions", "OpenLibm_jll", "OpenSpecFun_jll"]
git-tree-sha1 = "64cca0c26b4f31ba18f13f6c12af7c85f478cfde"
uuid = "276daf66-3868-5448-9aa4-cd146d93841b"
version = "2.5.0"

    [deps.SpecialFunctions.extensions]
    SpecialFunctionsChainRulesCoreExt = "ChainRulesCore"

    [deps.SpecialFunctions.weakdeps]
    ChainRulesCore = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"

[[deps.StaticArraysCore]]
git-tree-sha1 = "192954ef1208c7019899fbf8049e717f92959682"
uuid = "1e83bf80-4336-4d27-bf5d-d5a4f845583c"
version = "1.4.3"

[[deps.Statistics]]
deps = ["LinearAlgebra", "SparseArrays"]
uuid = "10745b16-79ce-11e8-11f9-7d13ad32a3b2"
version = "1.10.0"

[[deps.StringManipulation]]
deps = ["PrecompileTools"]
git-tree-sha1 = "725421ae8e530ec29bcbdddbe91ff8053421d023"
uuid = "892a3eda-7b42-436c-8928-eab12a02cf0e"
version = "0.4.1"

[[deps.StructTypes]]
deps = ["Dates", "UUIDs"]
git-tree-sha1 = "159331b30e94d7b11379037feeb9b690950cace8"
uuid = "856f2bd8-1eba-4b0a-8007-ebc267875bd4"
version = "1.11.0"

[[deps.SuiteSparse_jll]]
deps = ["Artifacts", "Libdl", "libblastrampoline_jll"]
uuid = "bea87d4a-7f5b-5778-9afe-8cc45184846c"
version = "7.2.1+1"

[[deps.TOML]]
deps = ["Dates"]
uuid = "fa267f1f-6049-4f14-aa54-33bafae1ed76"
version = "1.0.3"

[[deps.TableTraits]]
deps = ["IteratorInterfaceExtensions"]
git-tree-sha1 = "c06b2f539df1c6efa794486abfb6ed2022561a39"
uuid = "3783bdb8-4a98-5b6b-af9a-565f29a5fe9c"
version = "1.0.1"

[[deps.Tables]]
deps = ["DataAPI", "DataValueInterfaces", "IteratorInterfaceExtensions", "OrderedCollections", "TableTraits"]
git-tree-sha1 = "598cd7c1f68d1e205689b1c2fe65a9f85846f297"
uuid = "bd369af6-aec1-5ad0-b16a-f7cc5008161c"
version = "1.12.0"

[[deps.Test]]
deps = ["InteractiveUtils", "Logging", "Random", "Serialization"]
uuid = "8dfed614-e22c-5e08-85e1-65c5234f0b40"

[[deps.TranscodingStreams]]
git-tree-sha1 = "0c45878dcfdcfa8480052b6ab162cdd138781742"
uuid = "3bb67fe8-82b1-5028-8e26-92a6c54297fa"
version = "0.11.3"

[[deps.UUIDs]]
deps = ["Random", "SHA"]
uuid = "cf7118a7-6976-5b1a-9a39-7adc72f591a4"

[[deps.Unicode]]
uuid = "4ec0a83e-493e-50e2-b9ac-8f72acf5a8f5"

[[deps.WeakRefStrings]]
deps = ["DataAPI", "InlineStrings", "Parsers"]
git-tree-sha1 = "b1be2855ed9ed8eac54e5caff2afcdb442d52c23"
uuid = "ea10d353-3f73-51f8-a26c-33c1cb351aa5"
version = "1.4.2"

[[deps.WorkerUtilities]]
git-tree-sha1 = "cd1659ba0d57b71a464a29e64dbc67cfe83d54e7"
uuid = "76eceee3-57b5-4d4a-8e66-0e911cebbf60"
version = "1.6.1"

[[deps.Zlib_jll]]
deps = ["Libdl"]
uuid = "83775a58-1f1d-513f-b197-d71354ab007a"
version = "1.2.13+1"

[[deps.libblastrampoline_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "8e850b90-86db-534c-a0d3-1478176c7d93"
version = "5.8.0+1"
"""

# ╔═╡ Cell order:
# ╟─f4b6f9a1-ba6b-4072-8474-da83c35357f8
# ╟─db4365ba-9d00-4c52-b1a8-49b7698cd7bf
# ╠═983b81da-c743-11f0-1d86-f93134ca51df
# ╟─8f3e1c46-2b62-40d6-8f22-f7ecbb1a9d92
# ╠═f5b3c6fa-3fc7-4a73-a0b3-6e77cc0391f8
# ╟─24248c32-7214-48cc-8606-5235ca0c9682
# ╠═823fe90d-c475-4878-b40d-3bcd8d820f2d
# ╟─a3437289-87cc-4750-a4f0-2df5674b3e29
# ╠═e50e8537-775e-4846-ba3b-6488806a5ded
# ╟─201d4ec2-9c4b-4e56-89e7-ec98d19dfcad
# ╠═e1fe18c7-852f-45d1-bfc3-edf3d0a86046
# ╟─acffe2ea-e76e-4385-8084-57eb22847567
# ╠═b7271ab2-1cdd-4819-a6af-09addef53370
# ╟─8460c50c-9e78-48d6-926c-c5def0707f0e
# ╠═b6f4bd46-fcd7-4818-86d1-539092c4dd58
# ╟─c8b692cb-3ec7-42da-80d3-7459df1ae2c3
# ╠═aa368c66-a628-4266-9a9b-787139836bea
# ╟─cf3aa873-a855-45ce-9e67-6aea9e161268
# ╠═1566b406-89b8-4d63-9cf6-efa4f66b8574
# ╟─c354d05a-66b0-4b32-9414-140084ab3057
# ╠═3e9ac665-89fd-4ab0-a789-6a4e3700e722
# ╟─80a752e4-a209-44fd-a886-a727e215d340
# ╠═f58d5443-fa5e-4dcb-8ce4-e5991c135bd7
# ╠═bfddf2d6-6705-43f5-a159-80f1ba09444e
# ╟─c90b82df-a153-4b4e-8aa0-22fb11d57626
# ╠═391f1ec6-d519-4847-9a75-a6e51a416986
# ╟─67060b03-684b-4191-9119-65d6994faf64
# ╠═b1d278e8-6af7-4b8f-8fb8-061b1bc78d0a
# ╟─9ac8c40f-0625-4803-a5aa-8322d7051afe
# ╠═488b3b78-be30-4fa0-a3dd-08f3029f4a98
# ╟─f17de48c-8ac0-4637-94a4-5c9069525890
# ╠═ae036656-b16c-488e-b4d0-ebbad54fe5f0
# ╟─9bde5945-9777-4454-aceb-ddd645acc404
# ╠═256c5aeb-9bbf-416b-bce3-43bf7cdd08fb
# ╟─dbc5d90d-d6d1-4d72-95a3-1707a4fc3697
# ╟─26325d60-d99f-4887-a09b-0945940a8bf4
# ╠═3c3d9a5e-9bb1-46bb-bbb7-ca02f9e993ca
# ╠═8fe8336e-2a6f-4376-9881-34293303e97a
# ╟─7c074634-043a-4721-81f1-352d45c67ea2
# ╠═d6efb1a4-fd99-40e2-a4d2-11d006969750
# ╠═c567c1f2-1049-4285-a5f8-bc0d99b390da
# ╟─4e9c8d2d-a120-4982-858a-f6cc1743447a
# ╠═b9824aaa-0943-407d-9271-9792b263c340
# ╟─67e5cc65-2a07-4639-89b1-ab88b00658f6
# ╟─00000000-0000-0000-0000-000000000001
# ╟─00000000-0000-0000-0000-000000000002
