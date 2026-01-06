# Quick fix script to update app.py predictions tab with LSTM integration
import re

# Read the file
with open('app.py', 'r', encoding='utf-8') as f:
    content =f.read()

# Find and replace the predictions tab section
old_pred_section = '''    with pred_tab:
        st.markdown("### 🤖 GARCH Model Prediction")
        
        try:
            # Train GARCH model
            with st.spinner("Training GARCH model..."):
                forecaster = VolatilityForecaster()
                stock_df = pd.DataFrame(df_filtered)
                forecaster.train_model(selected_ticker, stock_df)
                prediction = forecaster.predict_volatility(selected_ticker, horizon=1)
            
            # Display prediction
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric(
                    "Predicted Volatility (Tomorrow)",
                    f"{prediction['predicted_volatility']:.2%}"
                )
            
            with col2:
                st.metric(
                    "Lower Bound (95% CI)",
                    f"{prediction['lower_bound']:.2%}"
                )
            
            with col3:
                st.metric(
                    "Upper Bound (95% CI)",
                    f"{prediction['upper_bound']:.2%}"
                )
            
            # Volatility interpretation
            if prediction['predicted_volatility'] > current_vol * 1.2:
                st.markdown("""
                <div class="alert-box alert-danger">
                    ⚠️ <strong>High Volatility Alert!</strong> Model predicts significant increase in volatility.
                </div>
                """, unsafe_allow_html=True)
            elif prediction['predicted_volatility'] < current_vol * 0.8:
                st.markdown("""
                <div class="alert-box alert-success">
                    ✓ <strong>Low Volatility Expected</strong> Model predicts calmer market conditions.
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown("""
                <div class="alert-box alert-warning">
                    → <strong>Stable Volatility</strong> Model predicts continuation of current patterns.
                </div>
                """, unsafe_allow_html=True)
            
            # Plot historical fit
            hist_vol = forecaster.get_historical_fit(selected_ticker)
            
            fig_pred = go.Figure()
            fig_pred.add_trace(go.Scatter(
                x=df_filtered.index[-len(hist_vol):],
                y=hist_vol,
                name='GARCH Fitted',
                line=dict(color='#00a8ff', width=2)
            ))
            fig_pred.add_trace(go.Scatter(
                x=df_filtered.index,
                y=df_filtered['Vol_GK'],
                name='Realized Volatility',
                line=dict(color='#00ff88', width=2, dash='dot')
            ))
            
            fig_pred.update_layout(
                title='GARCH Model Fit vs Realized Volatility',
                template='plotly_dark',
                paper_bgcolor='#0E1117',
                plot_bgcolor='#1c1f26',
                height=400,
                hovermode='x unified'
            )
            
            st.plotly_chart(fig_pred, use_container_width=True)
            
        except Exception as e:
            st.error(f"Error generating prediction: {str(e)}")'''

new_pred_section = '''    with pred_tab:
        # Load models and data
        lstm_model = load_lstm_model(selected_ticker)
        training_data = load_training_data(selected_ticker)
        sentiment_data = load_sentiment_data(selected_ticker)
        
        # Create tabs for different models
        garch_tab, lstm_tab, ensemble_tab, sentiment_tab = st.tabs([
            "📊 GARCH", "🧠 LSTM", "🎯 Ensemble", "📰 Sentiment"
        ])
        
        # ===== GARCH TAB =====
        with garch_tab:
            st.markdown("### GARCH(1,1) Volatility Forecast")
            
            try:
                with st.spinner("Training GARCH model..."):
                    forecaster = VolatilityForecaster()
                    stock_df = pd.DataFrame(df_filtered)
                    forecaster.train_model(selected_ticker, stock_df)
                    prediction = forecaster.predict_volatility(selected_ticker, horizon=1)
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Predicted Volatility", f"{prediction['predicted_volatility']:.2%}")
                with col2:
                    st.metric("Lower Bound (95%)", f"{prediction['lower_bound']:.2%}")
                with col3:
                    st.metric("Upper Bound (95%)", f"{prediction['upper_bound']:.2%}")
                
                hist_vol = forecaster.get_historical_fit(selected_ticker)
                
                fig_garch = go.Figure()
                fig_garch.add_trace(go.Scatter(
                    x=df_filtered.index[-len(hist_vol):],
                    y=hist_vol,
                    name='GARCH Fitted',
                    line=dict(color='#00a8ff', width=2)
                ))
                fig_garch.add_trace(go.Scatter(
                    x=df_filtered.index,
                    y=df_filtered['Vol_GK'],
                    name='Realized',
                    line=dict(color='#00ff88', width=2, dash='dot')
                ))
                
                fig_garch.update_layout(
                    title='GARCH Model Fit',
                    template='plotly_dark',
                    paper_bgcolor='#0E1117',
                    plot_bgcolor='#1c1f26',
                    height=400,
                    hovermode='x unified'
                )
                
                st.plotly_chart(fig_garch, use_container_width=True)
                
            except Exception as e:
                st.error(f"GARCH Error: {str(e)}")
        
        # ===== LSTM TAB =====
        with lstm_tab:
            st.markdown("### LSTM Deep Learning Forecast")
            
            if lstm_model is not None and training_data is not None:
                try:
                    # Prepare last sequence for prediction
                    X, _ = lstm_model.prepare_data(training_data.iloc[-100:], target_col='Vol_GK')
                    
                    if len(X) > 0:
                        lstm_pred = lstm_model.predict(X[-1:])
                        lstm_vol = lstm_pred[0]
                        
                        col1, col2 = st.columns(2)
                        with col1:
                            st.metric("LSTM Predicted Volatility", f"{lstm_vol:.2%}")
                        with col2:
                            diff = ((lstm_vol - current_vol) / current_vol * 100)
                            st.metric("vs Current", f"{diff:+.1f}%")
                        
                        # Show model info
                        st.info(f"✓ Using GPU-trained model ({len(X)} sequences, 10 features)")
                    else:
                        st.warning("Insufficient data for LSTM prediction")
                        
                except Exception as e:
                    st.error(f"LSTM Error: {str(e)}")
            else:
                st.warning("LSTM model not available for this stock")
        
        # ===== ENSEMBLE TAB =====
        with ensemble_tab:
            st.markdown("### 🎯 Ensemble Prediction (GARCH + LSTM)")
            
            try:
                # Get GARCH prediction
                forecaster = VolatilityForecaster()
                stock_df = pd.DataFrame(df_filtered)
                forecaster.train_model(selected_ticker, stock_df)
                garch_pred = forecaster.predict_volatility(selected_ticker, horizon=1)
                
                # Get LSTM prediction if available
                if lstm_model is not None and training_data is not None:
                    X, _ = lstm_model.prepare_data(training_data.iloc[-100:], target_col='Vol_GK')
                    if len(X) > 0:
                        lstm_vol = lstm_model.predict(X[-1:])[0]
                        
                        # Weighted ensemble (60% LSTM, 40% GARCH)
                        ensemble_vol = 0.6 * lstm_vol + 0.4 * garch_pred['predicted_volatility']
                        
                        col1, col2, col3, col4 = st.columns(4)
                        with col1:
                            st.metric("GARCH", f"{garch_pred['predicted_volatility']:.2%}")
                        with col2:
                            st.metric("LSTM", f"{lstm_vol:.2%}")
                        with col3:
                            st.metric("Ensemble", f"{ensemble_vol:.2%}", delta=f"{((ensemble_vol - current_vol) / current_vol * 100):+.1f}%")
                        with col4:
                            st.metric("Current", f"{current_vol:.2%}")
                        
                        # Alert
                        if ensemble_vol > current_vol * 1.2:
                            st.markdown('<div class="alert-box alert-danger">⚠️ <strong>High Volatility Alert!</strong></div>', unsafe_allow_html=True)
                        elif ensemble_vol < current_vol * 0.8:
                            st.markdown('<div class="alert-box alert-success">✓ <strong>Low Volatility Expected</strong></div>', unsafe_allow_html=True)
                        else:
                            st.markdown('<div class="alert-box alert-warning">→ <strong>Stable Volatility</strong></div>', unsafe_allow_html=True)
                    else:
                        st.info("Using GARCH only (insufficient LSTM data)")
                        st.metric("Predicted Volatility", f"{garch_pred['predicted_volatility']:.2%}")
                else:
                    st.info("Using GARCH only (LSTM not available)")
                    st.metric("Predicted Volatility", f"{garch_pred['predicted_volatility']:.2%}")
                    
            except Exception as e:
                st.error(f"Ensemble Error: {str(e)}")
        
        # ===== SENTIMENT TAB =====
        with sentiment_tab:
            if sentiment_data is not None and show_sentiment:
                st.markdown("### 📰 News Sentiment Analysis")
                
                # Display sentiment chart
                fig_sent = create_sentiment_chart(sentiment_data)
                st.plotly_chart(fig_sent, use_container_width=True)
                
                # Sentiment metrics
                col1, col2, col_3 = st.columns(3)
                with col1:
                    avg_sent = sentiment_data['sentiment_mean'].mean()
                    st.metric("Avg Sentiment (30d)", f"{avg_sent:.3f}")
                with col2:
                    latest_sent = sentiment_data['sentiment_mean'].iloc[-1]
                    st.metric("Latest Sentiment", f"{latest_sent:.3f}")
                with col_3:
                    total_news = sentiment_data['news_count'].sum()
                    st.metric("Total Articles", f"{int(total_news)}")
                    
            else:
                st.info("No sentiment data available for this stock")'''

# Replace
content = content.replace(old_pred_section, new_pred_section)

# Write back
with open('app.py', 'w', encoding='utf-8') as f:
    f.write(content)

print("✓ Dashboard integrated with LSTM and sentiment!")
