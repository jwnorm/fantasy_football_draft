### A Pluto.jl notebook ###
# v0.19.40

using Markdown
using InteractiveUtils

# ╔═╡ 3c75db1e-c7d1-11f0-30bb-6dc7f14d4063
using JuMP, HiGHS, CSV, DataFrames, OrderedCollections

# ╔═╡ ed6d51fc-3ad1-4864-935f-b0489d70536e
md"""
# Fantasy Football Draft - Scenario Analysis
**Jacob Norman**
"""

# ╔═╡ a5dcf5d4-22fd-4422-b7af-b1a77f08ce40
md"""
## Introduction

In the previous analysis, we succssfully modeled the 2025 NFL fantasy draft as a **B**inary **I**nteger **P**rogramming (BIP) model and compared it against the ideal roster through Week 13. This report will build on the previous analysis and will be concerned with the third objective: conducting a scenario analysis of different inputs, including:

* Projection system utilized
* Beginning draft position in the first round
* Proxy for player draft value
* Reception scoring settings
* Impact of removing the **T**ight **E**nd (TE) positional requirement
"""

# ╔═╡ 37b28b04-7d4a-48b9-8460-ade24b589718
md"""
## Preprocessing

To begin, we will load the libraries that we used in the previous document.
"""

# ╔═╡ fde70ca8-dcd9-48fa-87cc-c3f9bd5db92c
md"""
Next, we will define the list of  positions required on the 14-player roster and a lower bound on how many of each much be drafted. This is the same dictionary that we used previously.
"""

# ╔═╡ b03037f9-b5b1-4b71-94cd-a89f3af1b1cb
position_requirements = OrderedDict(
	"qb" => 1, 
	"rb" => 2, 
	"wr" => 2, 
	"te" => 1, 
	"flex" => 2
)

# ╔═╡ b28f74f5-7df1-40f9-b342-6bd2d2935928
md"""
To greatly reduce the amount of duplicate code we will need, let's create a program that will solve the BIP and allow use to adjust the inputs as needed. We can adapt two of the functions we created in the prior analysis for this purpose. Let's first create the helper function `solve_bip_model`.
"""

# ╔═╡ bba3ae72-5f6c-460d-a75f-d3ec144e099c
"""
This function solves a binary integer programming model using the `JuMP` library and `HiGHS` solver for the 2025 NFL fantasy football draft based on a series of user inputs.

Arguments

- `df::DataFrame`: DataFrame containing 2025 NFL fantasy point projections
- `M::Int64`: Number of players in player pool
- `N::Int64`: Number of rounds in draft
- `teams::Int64`: Number of teams in fantasy league
- `bye_weeks::Vector{String}`: Vector containing list of NFL bye weeks
- `position_requirements::OrderedDict{String, Int64}`: A dictionary of the roster positions and the minimum number of eligible players for each one
- `projections::Symbol`: Symbol for a column in `df` referring to a set of projections
- `draft_value::Symbol`: Symbol for a column in `df` referring to a draft value
- `no_tight_ends::Bool`: Whether to include tight ends in position requirements 
- `start_position::Int64`: Starting draft position in round one
- `verbose::Bool`: Print total projected points to screen

Returns   

- `x::Any`: Optimal x values from model results
"""
function solve_bip_model(df::DataFrame,
						 M::Int64,
						 N::Int64,
						 teams::Int64,
						 bye_weeks::Vector{String},
						 position_requirements::OrderedDict{String, Int64},
						 projections::Symbol,
					 	 draft_value::Symbol,
						 no_tight_ends::Bool,
						 start_position::Int64,
						 verbose::Bool)

	# initialize model
	model = Model(HiGHS.Optimizer)
	set_silent(model)

	# decision variables
	@variable(model, x[df[:, :player_name], 1:N], Bin)

	# objective function
	@objective(model, 
		   	   Max, 
		       sum(x[df[i, :player_name], j] * df[i, projections] 
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
				sum(df[i, draft_value] * x[df[i, :player_name], j] 
				for i=1:M) >= teams * (j - 1) + start_position)

	# draft position enforcement: even rounds
	if start_position == 1
		@constraint(model, 
					adp_even_max[j ∈ 2:2:N], 
					sum(df[i, draft_value] * x[df[i, :player_name], j] 
					for i=1:M) >= teams * j)
		
	elseif start_position == 6 
		@constraint(model, 
					adp_even_max[j ∈ 2:2:N], 
					sum(df[i, draft_value] * x[df[i, :player_name], j] 
					for i=1:M) >= teams * (j - 1) + start_position + 1)

	elseif start_position == 12	
		@constraint(model, 
					adp_even_max[j ∈ 2:2:N], 
					sum(df[i, draft_value] * x[df[i, :player_name], j] 
					for i=1:M) >= teams * (j - 2) + start_position + 1)
	end

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

# ╔═╡ f4b494f0-d4db-4127-9282-c5e6c6718458
md"""
Next, we will adapt `display_drafted_players` to work nicely with this new program.
"""

# ╔═╡ a4c3f96d-18ff-4d0c-99b0-5f60b194bc58
"""
This displays the output from a `JuMP` model for the fantasy football draft binary integer programming model as a `DataFrame`.

Arguments

- `x::Any`: Optimal x values from model results
- `N::Int64`: Number of rounds in draft
- `df::DataFrame`: DataFrame containing 2025 NFL fantasy point projections
- `projections::Symbol`: Symbol for a column in `df` referring to a set of projections
- `draft_value::Symbol`: Symbol for a column in `df` referring to a draft value

Returns   

- `clean_drafted_players::DataFrame`: Output of drafted players along with some descriptive statistics and their fantasy point totals.
"""
function display_drafted_players(x::Any, N::Int64, df::DataFrame, 
								 projections:: Symbol, draft_value::Symbol)

	# create copy of df
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
	cols = [:round, :player_name, :position, projections]
	clean_drafted_players = drafted_players_df[:, cols]
	
	return clean_drafted_players
	
end

# ╔═╡ a80af064-3091-4320-9acf-e82db0725902
md"""
Finally, we will stitch it all together in a main program, `run_fantasy_football_draft`, that will facilitate the scenario analysis.
"""

# ╔═╡ b257c46b-7983-416b-b316-2eb2e877ba43
"""
Full program for running a single binary integer programming model and outputting the results for the fantasy football draft problem.

Arguments

- `position_requirements::OrderedDict{String, Int64}`: A dictionary of the roster positions and the minimum number of eligible players for each one
- `projections::Symbol`, optional: Symbol for a column in `df` referring to a set of projections; default is `:athletic_ppr_projected_points`
- `draft_value::Symbol`, optional: Symbol for a column in `df` referring to a draft value; default is `:adp`
- `no_tight_ends::Bool`, optional: Whether to include tight ends in position requirements; default is `false`
- `start_position::Int64`, optional: Starting draft position in round one; default is `6`
- `verbose::Bool`, optional: Print total projected points to screen; default is `true`

Returns   

- `output::DataFrame`: Output of drafted players along with some descriptive statistics and their fantasy point totals.
"""
function run_fantasy_football_draft(
						 position_requirements::OrderedDict{String, Int64}; 
						 projections::Symbol=:athletic_ppr_projected_points,
					 	 draft_value::Symbol=:adp,
						 no_tight_ends::Bool=false,
						 start_position::Int64=6,
						 verbose::Bool=true)

	# read in data
	df = DataFrame(CSV.File("../data/fantasy_football_main_data.csv"))

	# sets
	M = length(df[:, :player_name])		# number of NFL players
	N = 14 								# number of rounds in draft

	# parameters
	teams = 12
	bye_weeks = ["5", "6", "7", "8", "9", "10", "11", "12", "14"]

	# remove tight end requirement
	pos_req = deepcopy(position_requirements)
	if no_tight_ends
		delete!(pos_req, "te")
	end 

	# solve bip model
	x = solve_bip_model(df, M, N, teams, bye_weeks, pos_req, projections, 
						draft_value, no_tight_ends, start_position, verbose)	

	# get output
	output = display_drafted_players(x, N, df, projections, draft_value)

	return output

end 

# ╔═╡ f60e3295-1d68-46a4-9c2d-3d779024d0c8
md"""
## Base Case

As a reminder, here are the parameters we defined for the base case:

* Scenario: Base
* Projections: *The Athletic*
* Scoring: PPR
* Draft Position: 6
* Draft Value: ADP
* Tight Ends: Yes

Let's test out our new program and confirm that it generates the same roster that we found previously.
"""

# ╔═╡ c99c5114-bb7f-4a9e-8535-43501a4fb6da
begin
	base_model = run_fantasy_football_draft(position_requirements)
	base_model
end

# ╔═╡ 6386a46e-d61e-4929-acff-baa703fe75b4
md"""
This is indeed the same roster and the same optimal value of total projected points. We will refer to this roster as the `base` roster throughout the remainder of this document.

Now, let's look at the model that is selected using actual fantasy points through Week 13 of the 2025 NFL season.
"""

# ╔═╡ 84cd04fb-c6cf-4b51-8978-669591a7bbee
begin
	true_model = run_fantasy_football_draft(position_requirements;
							   				projections=:week13_actual_ppr_points)
	true_model
end 

# ╔═╡ 33f25610-3729-4166-a335-97c5ce72e2d5
md"""
This `true` roster will be referenced and compared against the various scenarios as well. Let's create a function to calculate the actual point total for a drafted roster as a percentage of this **true** roster's points.
"""

# ╔═╡ efa15877-1686-4db5-b7cc-9952cf9d72a5
"""
Calculates the actual points scored through Week 13 for the given roster and that point total as a percentage of possible points based on the **true** roster of players.

Arguments

- `model_df::DataFrame`: A model dataframe returned from the  `run_fantasy_football_draft` program

Returns   

- `correct_pct::Float64`: Total actual points scored for drafted team / total actual points scored for **true** roster
"""
function calculate_point_attainment_percent(model_df::DataFrame)

	# read in data
	df = DataFrame(CSV.File("../data/fantasy_football_main_data.csv"))

	# get vector of drafted players
	drafted_players = model_df[:, :player_name]

	# filter df for drafted players only
	roster_df = filter(:player_name => row -> row ∈ drafted_players, df)

	# calculate point totals for drafted players and true roster
	drafted_total = sum(roster_df[:, :week13_actual_ppr_points])
	true_total = sum(true_model[:, :week13_actual_ppr_points])

	# calculate total point attainment percentage
	correct_pct = drafted_total / true_total

	print("Total Point Attainment Percent:\t", round(correct_pct * 100; digits=1), "%")

	return correct_pct
end

# ╔═╡ 44085f45-fcaf-4d5a-931f-55d9fb702114
md"""
Let's verify that this function calculates the same number we found in the last report.
"""

# ╔═╡ 292cc724-6e2f-4790-8929-de8e29e39e13
base_pct = calculate_point_attainment_percent(base_model);

# ╔═╡ a96c8e51-009c-4315-8b4a-f079ad0012f8
md"""
This is the same percentage. With that completed, we are ready to move on.
"""

# ╔═╡ a2fa6b2d-7a67-4dbb-94bf-d630d86c92e4
md"""
## Projection System

One of the primary factors that will influence the optimal solution is the projection system used to forecast total fantasy points. The BIP model is only as strong as the projections after all. This is why I chose to use [Jake Ciely's projections](https://www.nytimes.com/athletic/6432965/2025/06/19/fantasy-football-2025-rankings-projections-cheat-sheet/) from *The Athletic*, which I have utilized for several fantasy seasons and have a high degree of trust in. Other sources that I have obtained projections from include:

* **[Pro Football Focus (PFF)](https://www.pff.com/news/fantasy-football-rankings-builder-2025):** A football analytics website that specializes in advanced metrics and player grades.

* **[numberFire](https://www.fanduel.com/research/fantasy-football-printable-cheat-sheet-top-200-players-for-12-team-ppr-league-2025):** Partnered with FanDuel Sportsbook to provide daily fantasy sports statistics and forecasts.

* **[RotoBaller](https://www.rotoballer.com/free-fantasy-football-draft-cheat-sheet):** General fantasy sports website that offers fantasy analysis, player rankings, and other projections.

These models are proprietary, so there is not a ton of insight into the key differences between them. Instead, we will focus on the draft strategies that result from each model to see if there are any key differences. Additionally, the *best* projection system will be the one that has the highest attainment percentage compared against the **true** roster. To perform this operation, we will be adjusting the values of the coefficent $c_i$ for each player $i$.

> **Note:** If a player was missing a projected fantasy point total from a certain projection system, it was imputed as the average of the projections that had values for that same player. This impacted 71 of 832 player records (8.53 percent) across the four projection systems.
"""

# ╔═╡ 6c6317f5-36e3-4944-ac44-16ad89b82661
md"""
### PFF

To begin, let's use PFF projections.
"""

# ╔═╡ 5d07e897-ac1a-47d0-920c-37ec5690ba43
begin
	pff_model = run_fantasy_football_draft(position_requirements;
							   			   projections=:pff_ppr_projected_points)
	pff_model
end 

# ╔═╡ 193c967b-e0f6-44ff-9c53-b72606c70d27
md"""
Firstly, there is very little overlap between the `PFF` model and the `base` model: the only player that appears on both is Bucky Irving. The most important pick in any draft is often the first-round selection. The **PFF** model decided to go with Ashton Jeanty over Christian McCaffrey. Jeanty has no NFL experience, but there was considerable buzz about his potential coming into 2025; McCaffrey was viewed as risky because he is injury-prone and could see his performance regress coming off a major injury in 2024.

From a draft strategy perspective, the `PFF` model still values running backs by doubling up on running backs in the first two rounds; however, the strategy diverges considerably when it comes to quarterbacks. Instead of taking both quarterbacks back-to-back near the end of the draft, this model opts to select two premium options in the early rounds: Josh Allen and Patrick Mahomes. The quarterback position will be an area of strength for the `PFF` roster when compared to peer rosters resulting from the same draft.

Now, lets calculate the attainment percent.
"""

# ╔═╡ 09683eea-da3e-4106-b5c8-a0df0e2f94b0
	pff_pct = calculate_point_attainment_percent(pff_model);

# ╔═╡ 92d721fe-d483-44a4-8f20-757d5778ec62
md"""
This percentage is 5 percent lower than what we saw in the `base` case, suggesting that these projections are weaker than those created by *The Athletic*.
"""

# ╔═╡ 18d13fb4-b006-4123-a174-bafff9ca8aa3
md"""
### numberFire

Moving on, we will assess the numberFire projections next.
"""

# ╔═╡ 99dc336e-b4b4-4f8e-8e8e-f4a5e22588c8
begin
	numberfire_model = run_fantasy_football_draft(position_requirements;
							   			projections=:numberfire_ppr_projected_points)
	numberfire_model
end 

# ╔═╡ bfc56502-f8bb-4b47-8b0f-a895e88dfe05
md"""
The `numberFire` model follows a different strategy commonly known as *Hero RB*. A high value running back is taken in the first round, then no others are taken until the end of the draft. In this case, the hero running back is Christian McCaffrey. He is a unique case because he will catch considerably more balls than the average running back, giving him surplus value. The next several picks are all receivers. Intuitively, this strategy makes sense in a PPR league, since any reception is valued at a full point, and wide receivers are the primary pass-catching options. In fact, only one other running back is selected: Joe Mixon. Therefore, the running back position constraint is binding in this model. Going into 2025, it was expected that he may return from injury somewhere in the middle of the season. He has yet to return as of Week 13 and is likely to miss the entire year. With limited running back depth, this roster will face some significant challenges going forward. This was a high-risk, high-reward strategy that did not pay off in this case.

The quarterbacks were selected towards the end of this draft, in line with the common approach in most fantasy drafts. Two players in the `base` model repeat here: Christian McCaffrey and Travis Kelce. We are starting to see a trend that these players are great values at their draft positions.

We can now calculate the attainment percent for this model.
"""

# ╔═╡ 079eaefc-1f22-4844-b43f-e5eaa1fd8ef1
	numberfire_pct = calculate_point_attainment_percent(numberfire_model);

# ╔═╡ 93083aad-f12b-4976-9124-04ea2119ae77
md"""
This nearly equivalent to the `base` model, so we conclude that these projections are on par with the `base` model in terms of their predictive power.
"""

# ╔═╡ e7d3be37-622e-44c0-ba5e-47a321852e26
md"""
### RotoBaller

Moving to the last set of projections, let's see the roster determined by the RotoBaller forecast.
"""

# ╔═╡ a213ef1f-2525-4571-8208-106b0069c54a
begin
	rotoballer_model = run_fantasy_football_draft(position_requirements;
							   			projections=:rotoballer_ppr_projected_points)
	rotoballer_model
end 

# ╔═╡ 7d9fb468-8ced-4e89-beb9-de8619fb33aa
md"""
The `RotoBaller` model follows a balanced strategy, initially focusing on running backs and then shifting to wide receivers. In fact, the two running backs it selects are Christian McCaffrey and Jonathan Taylor, the top two players selected by the `true` model. This draft could almost be considered a *Hero RB* strategy as these are the only two running backs selected. They need to perform at elite levels because there is no depth to replace them; however, these are actually the top two overall players in all of fantasy through Week 13. It should be mentioned that the running back constraint is also binding here. 

Like in the `base` model, the quarterbacks are selected towards the end of the draft to emphasize the value of the other positions earlier in the draft. Another note is that this model selects two tight ends, positions that are typically more volatile in scoring and with lower ceilings.

Let's see what the attainment percentage is for this roster.
"""

# ╔═╡ d062b450-06af-4cd2-a9c7-6d5376250823
	rotoballer_pct = calculate_point_attainment_percent(rotoballer_model);

# ╔═╡ 3562bd2a-9d8c-4c2f-84c1-6793c047bb52
md"""
Wow, this is significantly higher than the other three models, suggesting that these are the strongest projections for the 2025 NFL season. This comes as no surpise since the top two players from the `true` roster are selected in the `RotoBaller` model: Christian McCaffrey and Jonathan Taylor. Since McCaffrey and Taylor are the overall number 1 and 2, respectively, fantasy players as of Week 13, which is driving a significant portion of this value.
"""

# ╔═╡ eb6dd48c-a72b-44ef-b55d-121d41ff8347
md"""
### Summary

Let's compare the aggregate statistics from the models discussed above in a summary `DataFrame`.
"""

# ╔═╡ 05a5c19a-03cd-4ac5-a9b4-d7c0d8e974f3
begin
	# get list of model names
	proj_models = ["true", "The Athletic (base)", "PFF", "numberFire", "RotoBaller"]

	# get list of model projected point totals
	proj_pts = [
		0,
		sum(base_model[!, :athletic_ppr_projected_points]),
		sum(pff_model[!, :pff_ppr_projected_points]),
		sum(numberfire_model[!, :numberfire_ppr_projected_points]),
		sum(rotoballer_model[!, :rotoballer_ppr_projected_points])]

	# get list of model attainment percentages
	proj_pcts = [1, base_pct, pff_pct, numberfire_pct, rotoballer_pct]

	# create and display output df
	DataFrame(models = proj_models, projected_points = proj_pts, percent=proj_pcts)
end

# ╔═╡ a7c5e1fd-eeba-46e3-90c0-ee67a275856c
md"""
We see that the `RotoBaller` model is the least optimistic in terms of projected points, while the `base` model is the most optimistic. The `base`, `PFF`, and `numberFire` models are all achieving around the same percentage of possible points through Week 13; however, the `RotoBaller` model is significantly higher, capturing over 70 percent of the points from the `true` optimal team. This seems like the best model to use going forward in future fantasy seasons.

Additionally, almost all models chose to draft running backs in the first two rounds and opted to select their quarterbacks towards the end of the draft. This indicates that, at least in 2025, these strategies are the preferred way to construct a roster. Lastly, certain players were routinely selected, affirming that the market undervalues these players relative to their expected fantasy points. This includes Christian McCaffrey, Bucky Irving, Tyreek Hill, DK Metcalf, Jakobi Meyers, and Travis Kelce.
"""

# ╔═╡ 3a63c7b0-ceb6-4cf6-a5fc-bb5352ea6aab
md"""
## Starting Draft Position

For the next topic, we will investigate whether the initial draft position has a material impact on roster construction. This will include the quality of players and the total projected fantasy points.

The first overall pick will be able to draft the top projected fantasy player; however, the snake format of the draft means that they will not draft another player until 23 picks later. Conversely, a team drafting in the last spot will be able to stack 2 players of the same approximate value in the first two rounds. Over the course of an entire draft, do these differences in draft position balance to be roughly equal, or is there an ideal starting position? Additionally, is there a distinct difference in approach depending on the starting position? We seek to determine the answer to these questions in this section.

The `base` model selects in the sixth position, which means that the team will be picking in the middle of each round.
"""

# ╔═╡ 3d3abc57-6882-4b37-9a16-e9aeb0dc05ab
md"""
### First Position

Let's start with looking at drafting in the first position.
"""

# ╔═╡ 9bc0fe08-4704-4ef8-9c57-de80e301dcbc
begin
	first_model = run_fantasy_football_draft(position_requirements;
											 start_position=1)
	first_model
end 

# ╔═╡ bbc5fe31-51d7-4e98-b42f-9e7cad561568
md"""
The early draft strategy certainly shifts when examining the `first` position model. This roster has a more balanced approach, selecting two running backs and two wide receivers in the first four rounds. This contrasts with the strong running back strategy observed when drafting sixth overall.

Many of the same players appear here compared to the `base` case, again indicating that they are a great value at their price. Christian McCaffrey has the highest projected point total among position players, so it seems this model will always opt to select him if possible. 

In the middle and later rounds, the strategy remains consistent. Quarterbacks are taken near the end, and two tight ends are drafted, all four of which are the same players selected in the **base** model.
"""

# ╔═╡ 2d082468-39ca-45b5-86ee-275ad6e58c35
md"""
### Twelfth Position

Let's now move to drafting at the end of the first round, in the twelfth position.
"""

# ╔═╡ 8b5f2990-e053-423b-9852-5aa9cae330f5
begin
	twelfth_model = run_fantasy_football_draft(position_requirements;
											   start_position=12)
	twelfth_model
end 

# ╔═╡ a82aeeb8-9aee-4e3f-8164-3e2a0ebebc40
md"""
Like in the other two cases, the `twelfth` position model balances the initial rounds equally with running backs and wide receivers. One notable difference is the philosophy around drafting a quarterback: Baker Mayfield is selected in the eighth round. This is a little earlier than in the `base` case, suggesting there is a falloff in value of quarterbacks after Caleb Williams. In other areas, this roster is quite similar to the previous models as well. Drafting in the `twelfth` position results in nearly 100 fewer projected fantasy points, a decrease in value of about 2.6 percent.
"""

# ╔═╡ 36b50ae3-b06b-446f-b992-4bd3940e68fb
md"""
### Summary
Let's review a summary of the projected point totals in the three scenarios as well as the excess value when compared to the `base` model.
"""

# ╔═╡ d27c8ad6-6cc8-4c4b-81ff-5d643fd7d9f0
begin
	# get list of model names
	pos_models = ["first", "sixth (base)", "twelfth"]

	# calculate point totals
	first_total = sum(first_model[!, :athletic_ppr_projected_points])
	sixth_total = sum(base_model[!, :athletic_ppr_projected_points])
	twelfth_total = sum(twelfth_model[!, :athletic_ppr_projected_points])

	# get list of model projected point totals
	pos_pts = [first_total, sixth_total, twelfth_total]

	# get list of excess values
	pos_pct = [(first_total / sixth_total) - 1, 0, (twelfth_total / sixth_total) - 1]

	# create and display output df
	DataFrame(models = pos_models, projected_points = pos_pts, excess_vaue = pos_pct)
end

# ╔═╡ 6534000e-2b9f-4c1e-ab50-6fd5d5d0e8cd
md"""
This is counterintuitive: drafting in the sixth position actually results in the largest increase in value relative to the other positions. The gap is between 50 and 100 points, which can be significant considering that individual weekly matchups can be decided by only a few points. The difference in points between drafting at the beginning or end of the first round is marginal, meaning a fantasy team owner should be agnostic between the two unless they feel strongly about a certain player.

The common thinking in the fantasy community is that drafting at the beginning or end of the first round would result in the most excess value (because you can stack picks back-to-back throughout the draft), but this is evidence that this is not the case. To determine whether this finding is an artifact of this specific set of projections, let's look at the actual total fantasy points scored through Week 13 by start position. This should eliminate any projection system bias.
"""

# ╔═╡ 631772f8-417f-488a-adaa-84a538662c81
begin
	# initialize empty list
	actual_point_totals = []

	# loop through each start position
	for p ∈ [1, 6, 12]

		# run model
		pmod = run_fantasy_football_draft(position_requirements; 
										  projections=:week13_actual_ppr_points,
										  start_position=p,
										  verbose=false);
	
			# get model point total
			pts = sum(pmod[!, :week13_actual_ppr_points])

			# append to list
			append!(actual_point_totals, pts)
		end

	# calculate excess values
	excess_vals = []
	for tot ∈ actual_point_totals
		append!(excess_vals, (tot / actual_point_totals[2]) - 1)
	end

	# create and display output df
	DataFrame(start_position = ["first", "sixth (base)", "twelfth"],
			  actual_points = actual_point_totals,
			  excess_value = excess_vals)
end 

# ╔═╡ ab3dc92b-bf9f-4f66-9472-4fdef8b5c5cf
md"""
Based on the actual results of the optimal fantasy team, it is clear that drafting in the middle of the first round is the most desirable option. It seems it is marginally better to draft in the twelfth position over the first; however, this may be specific to the 2025 season and not generalizable to all future fantasy years since the numbers are close.
"""

# ╔═╡ d5082e91-4464-4cad-9e6e-f148c1251d94
md"""
## Draft Value

The next parameter we will consider is draft value, which is what we are using to estimate when a player is normally selected in a draft. Up until this point, we have exclusively been using **A**verage **D**raft **P**osition (ADP), which is the average pick number a player is selected over a hundred or so fantasy drafts. [**N**ational **F**antasy **C**hampionship (NFC)](https://nfc.shgn.com/adp/football) also records the maximum and minimum pick that each player is selected among the drafts. The max pick attributes a lower value to each player, while the min pick overvalues them. While it is extremely unlikely that all players will be selected at their max/min pick, it would be helpful to quantify the potential difference in a team's fantasy total points with this in mind. Could this also result in changes to draft strategy?
"""

# ╔═╡ bfbf6471-46d9-48a8-82c9-e0c56ec3f0a4
md"""
### Maximum Draft Position

Initially, let's consider adjusting the player draft values to their max pick.
"""

# ╔═╡ c8aeede0-2697-4e75-9259-884383788e34
begin
	max_model = run_fantasy_football_draft(position_requirements;
										   draft_value=:max_draft_position)
	max_model
end 

# ╔═╡ 1de4f2c4-dc7d-4831-aaef-1a4333d63cb1
md"""
Giving a low value to each players frees up our model, allowing it a wider selection of players that were previously unavailable. This `max` model doubles down on prioritizing running backs, drafting five running backs in the first six rounds. The big addition here is De'von Achane, another pass-catching back in the same vein as Christian McCaffrey.

Quarterbacks and tight ends are taken considerably later in the draft when compared to the `base` model that uses ADP. This suggests that it is optimal to wait on these position groups as long as possible. This comes from experience and knowing the draft room you are in.

While a lot of the same players are selected, they are taken in later rounds than in the `base` model. This reminds us it is important to be aware of players sliding down the draft board and capitalize when it happens. In this case, Achane fell and the model took advantage.
"""

# ╔═╡ 02a25201-608b-49bf-a157-7665d49aad74
md"""
### Minimum Draft Position

Next, we will consider the min pick for each player.
"""

# ╔═╡ 8ea00307-d6fa-4b4a-9030-dd5d81f4a999
begin
	min_model = run_fantasy_football_draft(position_requirements;
										   draft_value=:min_draft_position)
	min_model
end 

# ╔═╡ 2ac8fc7c-5936-426a-83e5-2b9a8011b033
md"""
This `min` model is more restricted by assigning lower values to each player's draft position. We see that a lot of the position players are repeated, but they are drafted much earlier. This signifies that running backs and wide receivers are intregral to the success of any fantasy team and they should be prioritized. Additionally, the elite talent is missing from this roster; there is no single superstar who will clearly be the hero of the team.

The actual strategy is more balanced between running backs and wide receivers in the first several rounds. The position groups that are hurt the most here are quarterbacks and tight ends. All four of them are selected in the last four rounds, meaning the `min` model is getting the worst remaining players. This is especially detrimental with quarterbacks, since Tua Tagovailoa and Geno Smith are bottom third talents with large drops in projected fantasy points when compared to other models. Instead of reaching on earlier rounds for the better quarterbacks, this `min` model still asserts that waiting as long as possible on this position, no matter how dire it seems, is still the optimal strategy.
"""

# ╔═╡ 2d591b47-6242-47d8-8f5d-ff772f2a625c
md"""
### Summary

To wrap up this section, let's compare projected point totals and the change from the `base` model in a table.
"""

# ╔═╡ 8f5d13a7-3c51-4ef8-b917-0616d84a44bb
begin
	# get list of model names
	val_models = ["min", "avg (base)", "max"]

	# calculate point totals
	min_total = sum(min_model[!, :athletic_ppr_projected_points])
	avg_total = sum(base_model[!, :athletic_ppr_projected_points])
	max_total = sum(max_model[!, :athletic_ppr_projected_points])

	# get list of model projected point totals
	val_pts = [min_total, avg_total,max_total]

	# get list of excess values
	val_pct = [(min_total / avg_total) - 1, 0, (max_total / avg_total) - 1]

	# create and display output df
	DataFrame(models = val_models, projected_points = val_pts, excess_value=val_pct)
end

# ╔═╡ 7373690d-92c7-44e4-a242-0d645f9cc211
md"""
The projected point totals are directionally exactly what we might guess: highest for the `max` model and lowest for the `min` model; however, the insight here is that the benefit from lower player values is not as great as the cost for higher players values. This means even in the most optimistic draft where all players are undervalued by the other fantasy teams, the increase in your roster's total points is only about 5 percent. In other words, highly competitive leagues more heavily penalize overvaluing players rather than rewarding drafting undervalued players.

One other observation is that the draft position enforcement constraints are relaxed in the `max` model and restricted in the `min` model. This is because only the coefficients of the **R**ight-**H**and **S**ide (RHS) are adjusted.
"""

# ╔═╡ 7f758198-372a-450d-8f00-4b83298bf801
md"""
## Reception Scoring

One of the great things about fantasy leagues is that they are highly customizable, which includes how players are scored. There are many different options to consider, but often the biggest and most impactful choice is how to score a reception. There are three main settings for this value:

* **Point Per Reception (PPR):** Each reception a player makes is worth a full point. This is the new standard that has emerged in the last couple decades.

* **Half-PPR:** Each reception a player makes is worth half a point. More common in industry and high-stakes leagues; this is viewed as a more balanced scoring option.

* **Standard:** There are no points awarded for a player making a reception. This was the default option across most leagues for the first several decades of fantasy football, hence why it is called *standard* scoring.

Obviously, the less points that are scored for receptions, the less total fantasy points a drafted team will accumulate. All else equal, PPR projections will always be higher than half-point PPR projections, which will always be higher than standard scoring projections. For this reason, we will forgo analyzing total points scored for these scenarios.
"""

# ╔═╡ 725bf2a1-18b3-4471-9bb2-3ecc293d9d48
md"""
### Half-PPR

Looking at the same projections from *The Athletic*, let's reduce the points per reception to half and see how this impacts draft strategy and team composition.
"""

# ╔═╡ 2cbcb197-e696-4527-b2d1-2e23badd60f5
begin
	half_model = run_fantasy_football_draft(position_requirements;
										projections=:athletic_half_projected_points)
	half_model
end 

# ╔═╡ e5520263-a7c4-44d6-9ba6-570549bce81e
md"""
The `half`-point model chooses to select six running backs in a row to start the draft. The two starting flex positions will be occupied by running backs, meaning that four running backs will play each week. Many of the names we have seen before are included among these six, but some new additions are Omarion Hampton and Isiah Pacheco.

The `half` model does not completely give up on wide receivers, still selecting four of them in the middle to late rounds. It should also be pointed out that only one tight end is selected here, which means that their value is diminished enough that a roster should only carry one. In fact, he was even drafted in the final round.
"""

# ╔═╡ 7596ad4e-008c-4a45-82b8-7f2bfbd0158e
md"""
### Standard Scoring

Now we will completely remove the points attributed to receptions for a standard scoring model.
"""

# ╔═╡ 5c3285e6-987f-4412-a925-bc79bfc9914b
begin
	std_model = run_fantasy_football_draft(position_requirements;
										projections=:athletic_std_projected_points)
	std_model
end 

# ╔═╡ d52373a9-57d6-4d93-ba01-e523fd907720
md"""
Wow, the first eight rounds are all running backs and only two wide receivers are selected period. This `standard` model is indicating that receivers of any kind (including tight ends) have significantly less value compared to running backs. The wide receiver positional requirement has become binding in this model.

Of course, only four of these running backs will start, but this is also a position group that frequently suffers injuries that require prolonged absences. The deep bench will certainly supplement the roster in such cases. Additionally, one of these backs drafted in the middle rounds could end up being a sleeper pick, offering surplus value and supplanting a starter drafted earlier. Javonte Williams would fall into this category since he has been one of the top 10 running backs through Week 13 of the NFL season, yet he is taken in the eighth round.
"""

# ╔═╡ ce466b68-8c16-4aa6-b034-4826b2c468aa
md"""
### Summary

To review, the less points that are given to receptions, the more value is placed on running backs. This result seems apparent from the definition of the scoring. While the `half`-point model still attributes some value to receivers, the `standard` model eliminates most of this.

The curious thing is that even in a full-point PPR league, it is still optimal to stack running backs in the first two rounds. We observed this in the `base` model. No receiver can approach the value of the elite running backs in the 2025 scoring environment. An alternative interpretation is that *The Athletic's* projections may be biased towards running backs.
"""

# ╔═╡ 778c8dec-5c78-4a3d-9104-e586474e097f
md"""
## Tight End Requirement

There has been a growing movement in the fantasy community, especially among expert players, to eliminate the strict tight end requirement in favor of a TE/WR flexible position. The reason for this is twofold:

1) Tight end production is highly volatile. Even the elite players post game scores with near zero points several times per year.

2) There is a lack of quality tight end options. Generally speaking, there is a small handful of highly productive players, and then a steep drop-off to a pool of tight ends of approximately the same low value.

My hypothesis is that no tight ends are drafted if this constraint is removed. Let's see if this is the case.
"""

# ╔═╡ 9fee6849-4b11-4585-9adc-9fb54688cf33
begin
	te_model = run_fantasy_football_draft(position_requirements;
										  no_tight_ends=true)
	te_model
end 

# ╔═╡ 53eeb0f9-ab75-4c6e-b75e-61eee7b356f6
md"""
There is no change to the roster of the `base` case. We should have guessed this considering that this constraint is not binding; there are two tight ends drafted.

What about for any of the other projection systems, including actual PPR points through Week 13?
"""

# ╔═╡ ba2e0434-57a5-4cfc-9ac3-5eb9995d4703
begin
	# create dictionary of projection systems and col names
	proj = OrderedDict(
		"Athletic Half-PPR" => :athletic_half_projected_points, 
		"Athletic Standard PPR" => :athletic_std_projected_points, 
		"PFF PPR" => :pff_ppr_projected_points, 
		"numberFire PPR" => :numberfire_ppr_projected_points, 
		"RotoBaller PPR" => :rotoballer_ppr_projected_points, 
		"Week 13 Actual PPR" => :week13_actual_ppr_points
	)

	# initialize empty list
	number_of_tes = []

	# loop through each projection system
	for system ∈ keys(proj)

		# run model with no TE constraint
		tmod = run_fantasy_football_draft(position_requirements; 
										  projections=proj[system],
										  no_tight_ends=true,
										  verbose=false);

		# get number of TEs on roster
		te = count(x -> x == "TE", tmod[!, :position])

		# append to list
		append!(number_of_tes, te)
	end

	# create and display output df
	DataFrame(projection_system = collect(keys(proj)), 
			  number_of_tight_ends = number_of_tes)
end 

# ╔═╡ 070163ab-18f2-47e1-96b5-78a0d6377c8d
md"""
There is only one projection system, numberFire, that does not draft any tight ends when the requirement is removed. Even completely eliminating points per receptions does not remove tight ends from the optimal roster. This is evidence that our initial hypothesis is not correct. While the volatility of week-to-week tight end scoring can be frustrating, it appears that even an average tight end is preferable to a subpar wide receiver. If other fantasy players in a draft have the same intuition around tight ends, this is certainly a fact that can be exploited to improve the value of a roster. 

Creating a combination WR/TE position instead of TE-only does not reduce their value in a material way; however, it does offer enhanced roster flexibility in season. For example, if all of your tight ends have the same bye week, a wide receiver can be started in their place. This might not lead to more expected points that week, but it eliminates the need to drop a player and claim a tight end on the waiver wire.
"""

# ╔═╡ e298cc46-7a41-4f33-8672-6bb678d56b0a
md"""

## Summary

To wrap up, let's review the key takeaways that have been discovered by performing a scenario analysis on the five parameters.

**Projection System**\
The most powerful projection system, in terms of total possible points through Week 13, was *Rotoballer* by a large margin. Each system resulted in a different draft strategy in the early rounds, but all of them followed similar approaches to selecting quarterbacks and tight ends.

**Starting Draft Position:**\
There is evidence that drafting in the middle of the first round has an advantage compared to drafting in the beginning or end. Many of the same players are drafted in all three cases, the primary difference is how the first couple rounds are handled.

**Proxy for Draft Value**\
It might be self-evident that applying higher values to the entire player pool results in a weaker roster and applying lower values results in a stronger one; however, the percent change in both cases from the average case is not the same. The cost to roster value when players are overvalued is almost double compared to the benefit when players are undervalued; meaning that it is never a good idea to draft players earlier than expected. Another insight is that quarterbacks and tight ends have the lowest positional value among the all roster spots, so they are the first players that should be punted if things do not go as planned. This is reinforced with the fact that the high-value model selects bad quarterbacks late instead of good quarterbacks too early.

**Reception Scoring**\
Reducing the number of points scored for each reception has a significant impact to draft strategy and overall roster construction. Full-point PPR leagues have the most balance between running backs and wide receivers. Standard scoring leagues are greatly skewed towards running backs, both in terms of roster build and the first half of the fantasy draft. Half-point leagues are somewhere in the middle, giving higher weight to running backs, but keeping wide receivers relevant.

**Tight End Requirement**\
Removing a strict tight end roster spot in favor to a TE/WR flexible position does not generally alter draft strategy. There is still considerable value to be gained from drafting tight ends in the middle to late rounds over comparable wide receivers. The key benefit here comes during in-season roster management, such as bye week coverages and spot starts.

**Specific Players**\
Throughout all scenario analyses, certain players repeatedly were drafted and often in the same spots. This is strong evidence that these players are mispriced in the fantasy marketplace and shrewd managers should capitalize on the value they offer.

The unicorn running back Christian McCaffrey is the consensus first round pick for the 2025 NFL fantasy season. Other running backs that were routinely selected include: Bucky Irving, Alvin Kamara, and James Conner. The shared trait between these four players is that they are skilled at catching the ball in addition to rushing: a valuable skill to have in any PPR or half-PPR league. 

The quarterbacks that many models chose to draft are Baker Mayfield, Caleb Williams, and Justin Fields. These players are mobile and frequently use their legs to score rushing touchdowns, a key separator since a rushing touchdown is worth 50 percent more points compared to a passing touchdown. Wide receivers and tight ends that are commonly on model rosters include: Tyreek Hill, DK Metcalf, Jakobi Meyers, Cooper Kupp, Darnell Mooney, Jayden Higgins, Travis Kelce, and Hunter Henry. These receivers should be targeted as they are either coming off disappointing 2024 seasons or are on new teams in 2025.
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
# ╟─ed6d51fc-3ad1-4864-935f-b0489d70536e
# ╟─a5dcf5d4-22fd-4422-b7af-b1a77f08ce40
# ╟─37b28b04-7d4a-48b9-8460-ade24b589718
# ╠═3c75db1e-c7d1-11f0-30bb-6dc7f14d4063
# ╟─fde70ca8-dcd9-48fa-87cc-c3f9bd5db92c
# ╠═b03037f9-b5b1-4b71-94cd-a89f3af1b1cb
# ╟─b28f74f5-7df1-40f9-b342-6bd2d2935928
# ╠═bba3ae72-5f6c-460d-a75f-d3ec144e099c
# ╟─f4b494f0-d4db-4127-9282-c5e6c6718458
# ╠═a4c3f96d-18ff-4d0c-99b0-5f60b194bc58
# ╟─a80af064-3091-4320-9acf-e82db0725902
# ╠═b257c46b-7983-416b-b316-2eb2e877ba43
# ╟─f60e3295-1d68-46a4-9c2d-3d779024d0c8
# ╠═c99c5114-bb7f-4a9e-8535-43501a4fb6da
# ╟─6386a46e-d61e-4929-acff-baa703fe75b4
# ╠═84cd04fb-c6cf-4b51-8978-669591a7bbee
# ╟─33f25610-3729-4166-a335-97c5ce72e2d5
# ╠═efa15877-1686-4db5-b7cc-9952cf9d72a5
# ╟─44085f45-fcaf-4d5a-931f-55d9fb702114
# ╠═292cc724-6e2f-4790-8929-de8e29e39e13
# ╟─a96c8e51-009c-4315-8b4a-f079ad0012f8
# ╟─a2fa6b2d-7a67-4dbb-94bf-d630d86c92e4
# ╟─6c6317f5-36e3-4944-ac44-16ad89b82661
# ╠═5d07e897-ac1a-47d0-920c-37ec5690ba43
# ╟─193c967b-e0f6-44ff-9c53-b72606c70d27
# ╠═09683eea-da3e-4106-b5c8-a0df0e2f94b0
# ╟─92d721fe-d483-44a4-8f20-757d5778ec62
# ╟─18d13fb4-b006-4123-a174-bafff9ca8aa3
# ╠═99dc336e-b4b4-4f8e-8e8e-f4a5e22588c8
# ╟─bfc56502-f8bb-4b47-8b0f-a895e88dfe05
# ╠═079eaefc-1f22-4844-b43f-e5eaa1fd8ef1
# ╟─93083aad-f12b-4976-9124-04ea2119ae77
# ╟─e7d3be37-622e-44c0-ba5e-47a321852e26
# ╠═a213ef1f-2525-4571-8208-106b0069c54a
# ╟─7d9fb468-8ced-4e89-beb9-de8619fb33aa
# ╠═d062b450-06af-4cd2-a9c7-6d5376250823
# ╟─3562bd2a-9d8c-4c2f-84c1-6793c047bb52
# ╟─eb6dd48c-a72b-44ef-b55d-121d41ff8347
# ╠═05a5c19a-03cd-4ac5-a9b4-d7c0d8e974f3
# ╟─a7c5e1fd-eeba-46e3-90c0-ee67a275856c
# ╟─3a63c7b0-ceb6-4cf6-a5fc-bb5352ea6aab
# ╟─3d3abc57-6882-4b37-9a16-e9aeb0dc05ab
# ╠═9bc0fe08-4704-4ef8-9c57-de80e301dcbc
# ╟─bbc5fe31-51d7-4e98-b42f-9e7cad561568
# ╟─2d082468-39ca-45b5-86ee-275ad6e58c35
# ╠═8b5f2990-e053-423b-9852-5aa9cae330f5
# ╟─a82aeeb8-9aee-4e3f-8164-3e2a0ebebc40
# ╟─36b50ae3-b06b-446f-b992-4bd3940e68fb
# ╠═d27c8ad6-6cc8-4c4b-81ff-5d643fd7d9f0
# ╟─6534000e-2b9f-4c1e-ab50-6fd5d5d0e8cd
# ╠═631772f8-417f-488a-adaa-84a538662c81
# ╟─ab3dc92b-bf9f-4f66-9472-4fdef8b5c5cf
# ╟─d5082e91-4464-4cad-9e6e-f148c1251d94
# ╟─bfbf6471-46d9-48a8-82c9-e0c56ec3f0a4
# ╠═c8aeede0-2697-4e75-9259-884383788e34
# ╟─1de4f2c4-dc7d-4831-aaef-1a4333d63cb1
# ╟─02a25201-608b-49bf-a157-7665d49aad74
# ╠═8ea00307-d6fa-4b4a-9030-dd5d81f4a999
# ╟─2ac8fc7c-5936-426a-83e5-2b9a8011b033
# ╟─2d591b47-6242-47d8-8f5d-ff772f2a625c
# ╠═8f5d13a7-3c51-4ef8-b917-0616d84a44bb
# ╟─7373690d-92c7-44e4-a242-0d645f9cc211
# ╟─7f758198-372a-450d-8f00-4b83298bf801
# ╟─725bf2a1-18b3-4471-9bb2-3ecc293d9d48
# ╠═2cbcb197-e696-4527-b2d1-2e23badd60f5
# ╟─e5520263-a7c4-44d6-9ba6-570549bce81e
# ╟─7596ad4e-008c-4a45-82b8-7f2bfbd0158e
# ╠═5c3285e6-987f-4412-a925-bc79bfc9914b
# ╟─d52373a9-57d6-4d93-ba01-e523fd907720
# ╟─ce466b68-8c16-4aa6-b034-4826b2c468aa
# ╟─778c8dec-5c78-4a3d-9104-e586474e097f
# ╠═9fee6849-4b11-4585-9adc-9fb54688cf33
# ╟─53eeb0f9-ab75-4c6e-b75e-61eee7b356f6
# ╠═ba2e0434-57a5-4cfc-9ac3-5eb9995d4703
# ╟─070163ab-18f2-47e1-96b5-78a0d6377c8d
# ╟─e298cc46-7a41-4f33-8672-6bb678d56b0a
# ╟─00000000-0000-0000-0000-000000000001
# ╟─00000000-0000-0000-0000-000000000002
