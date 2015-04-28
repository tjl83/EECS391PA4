package edu.cwru.sepia.agent;

import edu.cwru.sepia.action.Action;
import edu.cwru.sepia.action.ActionFeedback;
import edu.cwru.sepia.action.ActionResult;
import edu.cwru.sepia.action.TargetedAction;
import edu.cwru.sepia.environment.model.history.DamageLog;
import edu.cwru.sepia.environment.model.history.DeathLog;
import edu.cwru.sepia.environment.model.history.History;
import edu.cwru.sepia.environment.model.state.State;
import edu.cwru.sepia.environment.model.state.Unit;
import edu.cwru.sepia.environment.model.state.Unit.UnitView;

import java.io.*;
import java.util.*;


public class RLAgent extends Agent {

    /**
	 * 
	 */
	private static final long serialVersionUID = 1L;
	/**
     * Set in the constructor. Defines how many learning episodes your agent should run for.
     * When starting an episode. If the count is greater than this value print a message
     * and call sys.exit(0)
     */
    public final int numEpisodes;
    
    /**
     * This is the number of learning episodes to run, set to 10 as was given by assignment
     */
    private final int numLearningEpisodes = 10;
    /**
     * This is the number of testing (evaluation) episodes to run, set to 5 as was given by assignment
     */
    private final int numTestingEpisodes = 5;
    /**
     * Total number of episodes played so far
     */
    private int episodesPlayed = 0;
    
    private enum Mode{
    	LEARNING,
    	TESTING;
    }
    
    /**
     * This returns whether the current episode is learning or testing the weights
     * @return The current episode mode, LEARNING or TESTING.
     */
    private Mode currentMode(){
    	// Learing happens between the number of learning episodes and after the number of testing episodes
    	// (e.g. with 10 learning: 0 - 9, 15 - 24, ...)
    	// Testing happens between the number of testing episodes and after the number of learning episodes
    	// (e.g. with 5 testing: 10 - 14, 25 - 29, ...)
    	
    	// The pattern repeats after every (#learning + #testing) episodes
    	int value = (this.episodesPlayed % (this.numLearningEpisodes + this.numTestingEpisodes));
    	
    	//  0-9 given 10 learning episodes
    	if(value < this.numLearningEpisodes)
    		return Mode.LEARNING;
    	// 10-14 given 5 testing episodes
    	else
    		return Mode.TESTING;
    }
    
    // Stores the cumulative reward for each test episode during testing
    private List<Double> testingRewards;
    // Stores the average cumulative reward for each set of testing episodes
    private List<Double> averageCumulativeRewards;
    // Maps each footman ID with its associated cummulative reward
    private Map<Integer, Double> footmenRewards;
    /**
     * List of your footmen and your enemies footmen
     */
    private List<Integer> myFootmen;
    private List<Integer> enemyFootmen;
    private Set<Integer> eliminatedEnemyFootmen;

    /**
     * Convenience variable specifying enemy agent number. Use this whenever referring
     * to the enemy agent. We will make sure it is set to the proper number when testing your code.
     */
    public static final int ENEMY_PLAYERNUM = 1;

    /**
     * Set this to whatever size your feature vector is.
     */
    public static final int NUM_FEATURES = 5;

    /** Use this random number generator for your epsilon exploration. When you submit we will
     * change this seed so make sure that your agent works for more than the default seed.
     */
    public final Random random = new Random(12345);

    /**
     * Your Q-function weights.
     */
    public Double[] weights;

    /**
     * These variables are set for you according to the assignment definition. You can change them,
     * but it is not recommended. If you do change them please let us know and explain your reasoning for
     * changing them.
     */
    public final double gamma = 0.9;
    public final double learningRate = .0001;
    public final double epsilon = .02;

    public RLAgent(int playernum, String[] args) {
        super(playernum);

        if (args.length >= 1) {
            numEpisodes = Integer.parseInt(args[0]);
            System.out.println("Running " + numEpisodes + " episodes.");
        } else {
            numEpisodes = 10;
            System.out.println("Warning! Number of episodes not specified. Defaulting to 10 episodes.");
        }

        boolean loadWeights = false;
        if (args.length >= 2) {
            loadWeights = Boolean.parseBoolean(args[1]);
        } else {
            System.out.println("Warning! Load weights argument not specified. Defaulting to not loading.");
        }

        if (loadWeights) {
            weights = loadWeights();
        } else {
            // initialize weights to random values between -1 and 1
            weights = new Double[NUM_FEATURES];
            for (int i = 0; i < weights.length; i++) {
                weights[i] = random.nextDouble() * 2 - 1;
            }
        }
    	testingRewards = new LinkedList<Double>();
    	averageCumulativeRewards = new LinkedList<Double>();
    }

    /**
     * We've implemented some setup code for your convenience. Change what you need to.
     */
    @Override
    public Map<Integer, Action> initialStep(State.StateView stateView, History.HistoryView historyView) {

        // You will need to add code to check if you are in a testing or learning episode
    	// This is handled with the method defined at the top of this file - currentMode()
    	
    	eliminatedEnemyFootmen = new HashSet<Integer>();
    	footmenRewards = new HashMap<Integer, Double>();

    	
        // Find all of your units
        myFootmen = new LinkedList<>();
        for (Integer unitId : stateView.getUnitIds(playernum)) {
            Unit.UnitView unit = stateView.getUnit(unitId);

            String unitName = unit.getTemplateView().getName().toLowerCase();
            if (unitName.equals("footman")) {
                myFootmen.add(unitId);
                footmenRewards.put(unitId, 0.0);
            } else {
                System.err.println("Unknown unit type: " + unitName);
            }
        }

        // Find all of the enemy units
        enemyFootmen = new LinkedList<>();
        for (Integer unitId : stateView.getUnitIds(ENEMY_PLAYERNUM)) {
            Unit.UnitView unit = stateView.getUnit(unitId);

            String unitName = unit.getTemplateView().getName().toLowerCase();
            if (unitName.equals("footman")) {
                enemyFootmen.add(unitId);
            } else {
                System.err.println("Unknown unit type: " + unitName);
            }
        }

        return middleStep(stateView, historyView);
    }

    /**
     * You will need to calculate the reward at each step and update your totals. You will also need to
     * check if an event has occurred. If it has then you will need to update your weights and select a new action.
     *
     * If you are using the footmen vectors you will also need to remove killed units. To do so use the historyView
     * to get a DeathLog. Each DeathLog tells you which player's unit died and the unit ID of the dead unit. To get
     * the deaths from the last turn do something similar to the following snippet. Please be aware that on the first
     * turn you should not call this as you will get nothing back.
     *
     * for(DeathLog deathLog : historyView.getDeathLogs(stateView.getTurnNumber() -1)) {
     *     System.out.println("Player: " + deathLog.getController() + " unit: " + deathLog.getDeadUnitID());
     * }
     *
     * You should also check for completed actions using the history view. Obviously you never want a footman just
     * sitting around doing nothing (the enemy certainly isn't going to stop attacking). So at the minimum you will
     * have an event occur whenever one your footmen's targets is killed or an action fails. Actions may fail if the target
     * is surrounded or the unit cannot find a path to the unit. To get the action results from the previous turn
     * you can do something similar to the following. Please be aware that on the first turn you should not call this
     *
     * Map<Integer, ActionResult> actionResults = historyView.getCommandFeedback(playernum, stateView.getTurnNumber() - 1);
     * for(ActionResult result : actionResults.values()) {
     *     System.out.println(result.toString());
     * }
     *
     * @return New actions to execute or nothing if an event has not occurred.
     */
    @Override
    public Map<Integer, Action> middleStep(State.StateView stateView, History.HistoryView historyView) {
    	updateFootmenRewards(stateView, historyView);
    	removeDeadUnits(stateView, historyView);
    	
    	Map<Integer, Action> actions = new HashMap<Integer, Action>();
    	
    	if(eventOccured(stateView, historyView)){
    		for(int attackerId : this.myFootmen){
    			int defenderId = selectAction(stateView, historyView, attackerId);
    			
    			if(currentMode().equals(Mode.LEARNING)){
    				Double[] oldWeights = this.weights;
    				double[] oldFeatures = this.calculateFeatureVector(stateView, historyView, attackerId, defenderId);
    				double totalReward = this.footmenRewards.get(attackerId);
    				this.weights = this.updateWeights(oldWeights, oldFeatures, totalReward, stateView, historyView, attackerId);
    			}
    			
    			actions.put(attackerId, Action.createCompoundAttack(attackerId, defenderId));
    		}
    	}
    	
        return actions;
    }
    
    /**
     * Determines if an event has occurred.
     * @param stateView
     * @param historyView
     * @return
     */
    private boolean eventOccured(State.StateView stateView, History.HistoryView historyView){
    	if(stateView.getTurnNumber() == 0)
        	// The episode begins
    		return true;
    	
    	int lastTurnNumber = stateView.getTurnNumber() - 1;
    	
    	List<DeathLog> deathLogs = historyView.getDeathLogs(lastTurnNumber);
    	if(deathLogs.size() > 0)
        	// If a death occurs...
    		return true;
    	
    	Map<Integer, ActionResult> actionResults = historyView.getCommandFeedback(this.playernum, lastTurnNumber);
    	for(ActionResult actionResult : actionResults.values()){
    		if(!actionResult.getFeedback().equals(ActionFeedback.INCOMPLETE))
        		// Either the action was completed or it failed if it isn't incomplete
    			return true;
    	}
    	
    	return false;
    }

    /**
     * Here you will calculate the cumulative average rewards for your testing episodes. If you have just
     * finished a set of test episodes you will call out testEpisode.
     *
     * It is also a good idea to save your weights with the saveWeights function.
     */
    @Override
    public void terminalStep(State.StateView stateView, History.HistoryView historyView) {
    	updateFootmenRewards(stateView, historyView);
    	removeDeadUnits(stateView, historyView);

		if (myFootmen.size() == 0) {
			System.out.println("You Lose. Enemy has " + enemyFootmen.size() + " footmen remaining");
		}
		else if (enemyFootmen.size() == 0) {
			System.out.println("You Win. You have " + myFootmen.size() + " footmen remaining");
		}
		else {
			System.err.println("ERROR: Winner unknown");
		}
		
    	switch(currentMode()){
	    	case LEARNING:
	    		System.out.println("Finished Learning Episode");
	    		
	    		break;
	    	case TESTING:
	    		System.out.println("Finished Testing Episode");
	    		Double sum = 0.0;
	    		for(Double reward : this.footmenRewards.values()){
	    			sum += reward;
	    		}
	    		this.testingRewards.add(sum);
	    		
	            // MAKE SURE YOU CALL printTestData after you finish a set of test episodes.
	    		if(this.testingRewards.size() == this.numTestingEpisodes){
	    			Double averageCumulativeReward = 0.0;
	    			for(Double cumulativeReward : testingRewards){
	    				averageCumulativeReward += cumulativeReward;
	    			}
	    			averageCumulativeReward /= (double)this.testingRewards.size();
	    			testingRewards = new LinkedList<Double>();
	    			averageCumulativeRewards.add(averageCumulativeReward);
		    		printTestData(averageCumulativeRewards);
	    		}
	    		break;
    		default:
    			break;
    	}
    	this.episodesPlayed++;
    	
    	if (this.episodesPlayed > numEpisodes){
    		System.out.println("Session Complete");
    		System.exit(0);
    	}
    	
        // Save your weights
        saveWeights(weights);

    }
    
    private void removeDeadUnits(State.StateView stateView, History.HistoryView historyView){
    	if(stateView.getTurnNumber() > 0){
    		int lastTurnNumber = stateView.getTurnNumber() - 1;
        	for(DeathLog deathLog : historyView.getDeathLogs(lastTurnNumber)) {
    			Integer unitId = new Integer(deathLog.getDeadUnitID());
        		if(deathLog.getController() == this.playernum){
        			this.myFootmen.remove(unitId);
        		}
        		else if(deathLog.getController() == this.ENEMY_PLAYERNUM){
        			this.enemyFootmen.remove(unitId);
        		}
        		else{
        			System.out.println("Phantom player?");
        		}
        	}
    	}
    }

    /**
     * Calculate the updated weights for this agent. 
     * @param oldWeights Weights prior to update
     * @param oldFeatures Features from (s,a)
     * @param totalReward Cumulative discounted reward for this footman.
     * @param stateView Current state of the game.
     * @param historyView History of the game up until this point
     * @param footmanId The footman we are updating the weights for
     * @return The updated weight vector.
     */
    public Double[] updateWeights(Double[] oldWeights, double[] oldFeatures, double totalReward, State.StateView stateView, History.HistoryView historyView, int footmanId) {
    	Double[] updatedWeightVector = new Double[oldWeights.length];

		double currentQVal = 0;
		for (int j = 0; j < oldFeatures.length; j++) {
			currentQVal += oldFeatures[j] * oldWeights[j];
		}

		double maxQVal = Double.NEGATIVE_INFINITY;
		for (Integer enemyFootmanId : enemyFootmen) {
			double qVal = calcQValue(stateView, historyView, footmanId, enemyFootmanId);
			if (qVal > maxQVal) {
				maxQVal = qVal;
			}
		}
		
		for (int i = 0; i < oldWeights.length; i++) {
			double targetQVal = totalReward + gamma * maxQVal;
			double dldw = -1 * (targetQVal - currentQVal) * oldFeatures[i];
			updatedWeightVector[i] = oldWeights[i] - learningRate * (dldw);
		}
		return updatedWeightVector;
    }

    /**
     * Given a footman and the current state and history of the game select the enemy that this unit should
     * attack. This is where you would do the epsilon-greedy action selection.
     *
     * @param stateView Current state of the game
     * @param historyView The entire history of this episode
     * @param attackerId The footman that will be attacking
     * @return The enemy footman ID this unit should attack
     */
    public int selectAction(State.StateView stateView, History.HistoryView historyView, int attackerId) {
    	int defenderId = -1;
    	
    	switch(this.currentMode()){
	    	case LEARNING:
	    		// Execute a random action with probability epsilon
	    		if(random.nextDouble() < epsilon){
					int index = (int) random.nextDouble() * enemyFootmen.size();
					defenderId = this.enemyFootmen.get(index);
	    			break;
	    		}
	    		// Otherwise follow the action recommended by the current policy
	    	case TESTING:
	    		double currQValue;
		    	double maxQValue = Double.MIN_VALUE;
		    	for(int possibleDefenderId : this.enemyFootmen){
		    		currQValue = calcQValue(stateView, historyView, attackerId, possibleDefenderId);
		    		if(currQValue > maxQValue){
		    			maxQValue = currQValue;
		    			defenderId = possibleDefenderId;
		    		}
		    	}
				break;
    		default:
    			System.out.println("No enemies left to attack");
    			break;
    	}
    	
    	return defenderId;
    }
    
    /**
     * Updates the rewards for each footman.
     * @param stateView
     * @param historyView
     */
    private void updateFootmenRewards(State.StateView stateView, History.HistoryView historyView){
    	if(stateView.getTurnNumber() > 0){
    		Iterator<Integer> iter = this.myFootmen.iterator();
    		Integer footmanId;
    		while(iter.hasNext()){
    			footmanId = iter.next();

	    		double currentReward = calculateReward(stateView, historyView, footmanId);
	    		double oldCumulativeReward = this.footmenRewards.get(footmanId);
	    		double newCumulativeReward = oldCumulativeReward + currentReward;
	    		this.footmenRewards.put(footmanId, newCumulativeReward);
	    		
    		}
    	}
    }

    /**
     * Given the current state and the footman in question calculate the reward received on the last turn.
     * This is where you will check for things like Did this footman take or give damage? Did this footman die
     * or kill its enemy. Did this footman start an action on the last turn? See the assignment description
     * for the full list of rewards.
     *
     * Remember that you will need to discount this reward based on the timestep it is received on. See
     * the assignment description for more details.
     *
     * As part of the reward you will need to calculate if any of the units have taken damage. You can use
     * the history view to get a list of damages dealt in the previous turn. Use something like the following.
     *
     * for(DamageLog damageLogs : historyView.getDamageLogs(lastTurnNumber)) {
     *     System.out.println("Defending player: " + damageLog.getDefenderController() + " defending unit: " + \
     *     damageLog.getDefenderID() + " attacking player: " + damageLog.getAttackerController() + \
     *     "attacking unit: " + damageLog.getAttackerID());
     * }
     *
     * You will do something similar for the deaths. See the middle step documentation for a snippet
     * showing how to use the deathLogs.
     *
     * To see if a command was issued you can check the commands issued log.
     *
     * Map<Integer, Action> commandsIssued = historyView.getCommandsIssued(playernum, lastTurnNumber);
     * for (Map.Entry<Integer, Action> commandEntry : commandsIssued.entrySet()) {
     *     System.out.println("Unit " + commandEntry.getKey() + " was command to " + commandEntry.getValue().toString);
     * }
     *
     * @param stateView The current state of the game.
     * @param historyView History of the episode up until this turn.
     * @param footmanId The footman ID you are looking for the reward from.
     * @return The current reward
     */
    public double calculateReward(State.StateView stateView, History.HistoryView historyView, int footmanId) {
    	double reward = 0.0;
    	
    	int lastTurnNumber = stateView.getTurnNumber() - 1;

    	for(DamageLog damageLog : historyView.getDamageLogs(lastTurnNumber)) {
    		if(damageLog.getAttackerID() == footmanId){
    			reward += damageLog.getDamage();
    		}
    		if(damageLog.getDefenderID() == footmanId){
    			reward -= damageLog.getDamage();
    		}
    	}
    	
    	for(DeathLog deathLog : historyView.getDeathLogs(lastTurnNumber)) {
    		// If this footman has died, 
    		if(deathLog.getDeadUnitID() == footmanId){
    			reward -= 100;
    		}
    		// If the reward for killing it's targeted enemy is already claimed, this footman cannot claim it.
    		if(deathLog.getController() == this.ENEMY_PLAYERNUM){
    			Map<Integer, ActionResult> actionResults = historyView.getCommandFeedback(this.playernum, lastTurnNumber);
    			if(actionResults.containsKey(new Integer(footmanId))){
    				TargetedAction action = (TargetedAction) actionResults.get(footmanId).getAction();
    				Integer defenderId = action.getTargetId();
            		if(	!this.eliminatedEnemyFootmen.contains(new Integer(defenderId)) &&
            				deathLog.getDeadUnitID() == defenderId) {
                			this.eliminatedEnemyFootmen.add(new Integer(defenderId));
                			reward += 100;
            		}
    			}
    		}
    	}

    	// If a command was issued last turn, check if the same command was issued the turn before (an attack targeting the same target)
    	Map<Integer, Action> commandsIssued = historyView.getCommandsIssued(playernum, lastTurnNumber);
    	Map<Integer, Action> commandsIssuedBeforeLast = historyView.getCommandsIssued(playernum, lastTurnNumber - 1);
    	if(commandsIssued.containsKey(footmanId) && commandsIssuedBeforeLast.containsKey(footmanId)){
    		TargetedAction commandIssued = (TargetedAction) commandsIssued.get(footmanId);
    		int enemyFootmanId = commandIssued.getTargetId();
    		TargetedAction commandIssuedBeforeLast = (TargetedAction) commandsIssued.get(footmanId);
    		int enemyFootmanIdBeforeLast = commandIssuedBeforeLast.getTargetId();
    		
    		if(enemyFootmanId != enemyFootmanIdBeforeLast)
    			reward -= 0.1;
    	}
    	
        return reward;
    }

    /**
     * Calculate the Q-Value for a given state action pair. The state in this scenario is the current
     * state view and the history of this episode. The action is the attacker and the enemy pair for the
     * SEPIA attack action.
     *
     * This returns the Q-value according to your feature approximation. This is where you will calculate
     * your features and multiply them by your current weights to get the approximate Q-value.
     *
     * @param stateView Current SEPIA state
     * @param historyView Episode history up to this point in the game
     * @param attackerId Your footman. The one doing the attacking.
     * @param defenderId An enemy footman that your footman would be attacking
     * @return The approximate Q-value
     */
    public double calcQValue(State.StateView stateView,
                             History.HistoryView historyView,
                             int attackerId,
                             int defenderId) {
    	double qValue = 0.0;
    	double[] featureVector = calculateFeatureVector(stateView, historyView, attackerId, defenderId);
    	
    	for(int i = 0; i < featureVector.length; i++){
    		qValue += this.weights[i] * featureVector[i];
    	}
    	
        return qValue;
    }

    /**
     * Given a state and action calculate your features here. Please include a comment explaining what features
     * you chose and why you chose them.
     *
     * All of your feature functions should evaluate to a double. Collect all of these into an array. You will
     * take a dot product of this array with the weights array to get a Q-value for a given state action.
     *
     * It is a good idea to make the first value in your array a constant. This just helps remove any offset
     * from 0 in the Q-function. The other features are up to you. Many are suggested in the assignment
     * description.
     *
     * @param stateView Current state of the SEPIA game
     * @param historyView History of the game up until this turn
     * @param attackerId Your footman. The one doing the attacking.
     * @param defenderId An enemy footman. The one you are considering attacking.
     * @return The array of feature function outputs.
     */
    public double[] calculateFeatureVector(State.StateView stateView,
                                           History.HistoryView historyView,
                                           int attackerId,
                                           int defenderId) {
    	double[] featureVector = new double[RLAgent.NUM_FEATURES];
    	
    	double constant = 0.0;
    	featureVector[0] = constant;
    	
    	double distanceFeature = calculateDistanceFeature(stateView, historyView, attackerId, defenderId);
    	featureVector[1] = distanceFeature;
    	double healthFeature = calculateHealthFeature(stateView, historyView, attackerId, defenderId);
    	featureVector[2] = healthFeature;
    	double numbersFeature = calculateNumbersFeature(stateView, historyView, attackerId, defenderId);
    	featureVector[3] = numbersFeature;
    	double isAttackingSelfFeature = calculateIsAttackingSelfFeature(stateView, historyView, attackerId, defenderId);
    	featureVector[4] = isAttackingSelfFeature;
    	
    	
    	return featureVector;
    }
    
    /**
     * This returns the feature representing the Chebyshev Distance between the ally footman and the enemy footman.
     * 
     * @param stateView Current state of the SEPIA game
     * @param historyView History of the game up until this turn
     * @param attackerId Your footman. The one doing the attacking.
     * @param defenderId An enemy footman. The one you are considering attacking.
     * @return The distance feature to be used in calculating the Q-Value.
     */
    private double calculateDistanceFeature(State.StateView stateView,
                                           History.HistoryView historyView,
                                           int attackerId,
                                           int defenderId) {
    	double distanceFeature = 0.0;
    	
    	UnitView attacker = stateView.getUnit(attackerId);
    	UnitView defender = stateView.getUnit(defenderId);
    	
    	int targetDistance = chebyshevDistance(attacker.getXPosition(), attacker.getYPosition(), defender.getXPosition(), defender.getYPosition());

		int numberOfEnemiesCloser = 0;
		for (int enemyID : enemyFootmen) {
			UnitView enemy = stateView.getUnit(enemyID);
			int distance = chebyshevDistance(attacker.getXPosition(),
					attacker.getYPosition(), enemy.getXPosition(), enemy.getYPosition());
			if (distance < targetDistance) {
				numberOfEnemiesCloser++;
			}
		}
		distanceFeature = enemyFootmen.size() - numberOfEnemiesCloser;
		
    	return distanceFeature;
    }
    
    private int chebyshevDistance(int x1, int y1, int x2, int y2){
    	return Math.max(Math.abs(x1 - x2), Math.abs(y1 - y2));
    }
    
    /**
     * This returns the feature representing the health ratio between the ally footman and the enemy footman.
     * 
     * @param stateView Current state of the SEPIA game
     * @param historyView History of the game up until this turn
     * @param attackerId Your footman. The one doing the attacking.
     * @param defenderId An enemy footman. The one you are considering attacking.
     * @return The health feature to be used in calculating the Q-Value.
     */
    private double calculateHealthFeature(State.StateView stateView,
                                           History.HistoryView historyView,
                                           int attackerId,
                                           int defenderId) {
    	double healthFeature = 0.0;
    	
    	UnitView attacker = stateView.getUnit(attackerId);
    	UnitView defender = stateView.getUnit(defenderId);
    	
    	healthFeature = ((double)attacker.getHP()) / ((double)defender.getHP());
    	
    	return healthFeature;
    }
    
    /**
     * This returns the feature representing the number of ally footmen currently attacking your footman's desginated target enemy footman
     * in ratio to your remaining footmen.
     * 
     * @param stateView Current state of the SEPIA game
     * @param historyView History of the game up until this turn
     * @param attackerId Your footman. The one doing the attacking.
     * @param defenderId An enemy footman. The one you are considering attacking.
     * @return The numbers feature to be used in calculating the Q-Value.
     */
    private double calculateNumbersFeature(State.StateView stateView,
                                           History.HistoryView historyView,
                                           int attackerId,
                                           int defenderId) {
    	double numbersFeature = 0.0;

    	int lastTurnNumber = stateView.getTurnNumber() - 1;
    	Map<Integer, Action> commandsIssued = historyView.getCommandsIssued(RLAgent.ENEMY_PLAYERNUM, lastTurnNumber);
    	
    	for(Integer myFootman:this.myFootmen){
    		if(commandsIssued.containsKey(myFootman)){
    			TargetedAction action = (TargetedAction) commandsIssued.get(myFootman);
    			if(action.getTargetId() == defenderId){
    				numbersFeature += 1.0;
    			}
    		}
    	}
    	
    	numbersFeature /= this.myFootmen.size();
    	
    	return numbersFeature;
    }
    
    /**
     * This returns the feature representing whether or not your footman's designated target enemy footman is currently attacking your footman.
     * 
     * @param stateView Current state of the SEPIA game
     * @param historyView History of the game up until this turn
     * @param attackerId Your footman. The one doing the attacking.
     * @param defenderId An enemy footman. The one you are considering attacking.
     * @return The feature, representing whether or not the target enemy is targeting your ally, to be used in calculating the Q-Value.
     */
    private double calculateIsAttackingSelfFeature(State.StateView stateView,
                                           History.HistoryView historyView,
                                           int attackerId,
                                           int defenderId) {
    	double isAttackingSelfFeature = 0.0;

    	int lastTurnNumber = stateView.getTurnNumber() - 1;
    	Map<Integer, Action> commandsIssued = historyView.getCommandsIssued(RLAgent.ENEMY_PLAYERNUM, lastTurnNumber);
    	if(commandsIssued.containsKey(defenderId)){
        	TargetedAction defenderAction = (TargetedAction) commandsIssued.get(defenderId);
        	if(defenderAction.getTargetId() == attackerId)
        		isAttackingSelfFeature = 1.0;
    	}
    	
    	return isAttackingSelfFeature;
    }

    /**
     * DO NOT CHANGE THIS!
     *
     * Prints the learning rate data described in the assignment. Do not modify this method.
     *
     * @param averageRewards List of cumulative average rewards from test episodes.
     */
    public void printTestData (List<Double> averageRewards) {
        System.out.println("");
        System.out.println("Games Played      Average Cumulative Reward");
        System.out.println("-------------     -------------------------");
        for (int i = 0; i < averageRewards.size(); i++) {
            String gamesPlayed = Integer.toString(10*i);
            String averageReward = String.format("%.2f", averageRewards.get(i));

            int numSpaces = "-------------     ".length() - gamesPlayed.length();
            StringBuffer spaceBuffer = new StringBuffer(numSpaces);
            for (int j = 0; j < numSpaces; j++) {
                spaceBuffer.append(" ");
            }
            System.out.println(gamesPlayed + spaceBuffer.toString() + averageReward);
        }
        System.out.println("");
    }

    /**
     * DO NOT CHANGE THIS!
     *
     * This function will take your set of weights and save them to a file. Overwriting whatever file is
     * currently there. You will use this when training your agents. You will include th output of this function
     * from your trained agent with your submission.
     *
     * Look in the agent_weights folder for the output.
     *
     * @param weights Array of weights
     */
    public void saveWeights(Double[] weights) {
        File path = new File("agent_weights/weights.txt");
        // create the directories if they do not already exist
        path.getAbsoluteFile().getParentFile().mkdirs();

        try {
            // open a new file writer. Set append to false
            BufferedWriter writer = new BufferedWriter(new FileWriter(path, false));

            for (double weight : weights) {
                writer.write(String.format("%f\n", weight));
            }
            writer.flush();
            writer.close();
        } catch(IOException ex) {
            System.err.println("Failed to write weights to file. Reason: " + ex.getMessage());
        }
    }

    /**
     * DO NOT CHANGE THIS!
     *
     * This function will load the weights stored at agent_weights/weights.txt. The contents of this file
     * can be created using the saveWeights function. You will use this function if the load weights argument
     * of the agent is set to 1.
     *
     * @return The array of weights
     */
    public Double[] loadWeights() {
        File path = new File("agent_weights/weights.txt");
        if (!path.exists()) {
            System.err.println("Failed to load weights. File does not exist");
            return null;
        }

        try {
            BufferedReader reader = new BufferedReader(new FileReader(path));
            String line;
            List<Double> weights = new LinkedList<>();
            while((line = reader.readLine()) != null) {
                weights.add(Double.parseDouble(line));
            }
            reader.close();

            return weights.toArray(new Double[weights.size()]);
        } catch(IOException ex) {
            System.err.println("Failed to load weights from file. Reason: " + ex.getMessage());
        }
        return null;
    }

    @Override
    public void savePlayerData(OutputStream outputStream) {

    }

    @Override
    public void loadPlayerData(InputStream inputStream) {

    }
}
