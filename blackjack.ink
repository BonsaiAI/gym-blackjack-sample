schema GameState
    Int8{0:31} current_sum,
    Int8{0:10} dealer_card,
    Int8{0, 1} usable_ace
end

schema Action
    Int8{0, 1} command
end

schema BlackJackConfig
    Int8 episode_length,
    Int8 num_episodes,
    UInt8 deque_size
end

simulator blackjack_simulator(BlackJackConfig)
    action (Action)
    state (GameState)
end

concept high_score is classifier
    predicts (Action)
    follows input(GameState)
    feeds output
end

curriculum win_curriculum
    train high_score
    with simulator blackjack_simulator
    objective open_ai_gym_default_objective

        lesson win
            configure
                constrain episode_length with Int8{-1},
                constrain num_episodes with Int8{-1},
                constrain deque_size with UInt8{1}
            until
                maximize open_ai_gym_default_objective
end
