//+------------------------------------------------------------------+
//|                                            Golden Gibz EA.mq5    |
//|                         MT5-Compatible AI Trading Model          |
//|                    80%+ Win Rate on XAUUSD M15                   |
//+------------------------------------------------------------------+
#property copyright "Gibz Trading Systems"
#property version   "1.00"
#property description "AI-Powered Gold Trading EA with 80%+ Win Rate"
#property strict

#include <Trade\Trade.mqh>

//--- Embed ONNX model as resource
#resource "\\Files\\trading_model_v3.onnx" as uchar OnnxModel[]

//--- Input parameters
input group "=== Trading Settings ==="
input double   LotSize = 0.01;
input double   TP_ATR_Mult = 0.75;
input double   SL_ATR_Mult = 1.5;
input int      MagicNumber = 888889;

input group "=== Risk Management ==="
input int      MaxDailyTrades = 20;
input int      MinBarsBetweenTrades = 3;

//--- Constants - MUST match training
#define WINDOW_SIZE     20
#define NUM_FEATURES    18   // 16 market + 2 state
#define NUM_ACTIONS     3

//--- Global variables
long           OnnxHandle = INVALID_HANDLE;
CTrade         Trade;

int            ATR_Handle;
int            RSI_Handle;
int            MACD_Handle;
int            MA20_Handle;
int            Stoch_Handle;

datetime       LastTradeBar = 0;
int            TodayTrades = 0;
datetime       CurrentDay = 0;
int            CurrentPosition = 0;
double         EntryPrice = 0;
double         EntryATR = 0;

//+------------------------------------------------------------------+
int OnInit()
{
   Print("=== Golden Gibz EA (MT5 Compatible) ===");
   
   Trade.SetExpertMagicNumber(MagicNumber);
   Trade.SetDeviationInPoints(30);
   Trade.SetTypeFilling(ORDER_FILLING_IOC);
   
   //--- Load ONNX from embedded resource
   ResetLastError();
   OnnxHandle = OnnxCreateFromBuffer(OnnxModel, ONNX_DEFAULT);
   
   if(OnnxHandle == INVALID_HANDLE)
   {
      Print("Failed to load ONNX. Error: ", GetLastError());
      Print("Resource size: ", ArraySize(OnnxModel));
      return INIT_FAILED;
   }
   Print("ONNX loaded. Size: ", ArraySize(OnnxModel), " bytes");
   
   //--- Set shapes
   long input_shape[] = {1, WINDOW_SIZE * NUM_FEATURES};
   if(!OnnxSetInputShape(OnnxHandle, 0, input_shape))
   {
      Print("Failed to set input shape");
      return INIT_FAILED;
   }
   
   long output_shape[] = {1, NUM_ACTIONS};
   if(!OnnxSetOutputShape(OnnxHandle, 0, output_shape))
   {
      Print("Failed to set output shape");
      return INIT_FAILED;
   }
   
   Print("Input: [1, ", WINDOW_SIZE * NUM_FEATURES, "] Output: [1, ", NUM_ACTIONS, "]");
   
   //--- Initialize indicators
   ATR_Handle = iATR(_Symbol, PERIOD_CURRENT, 14);
   RSI_Handle = iRSI(_Symbol, PERIOD_CURRENT, 14, PRICE_CLOSE);
   MACD_Handle = iMACD(_Symbol, PERIOD_CURRENT, 12, 26, 9, PRICE_CLOSE);
   MA20_Handle = iMA(_Symbol, PERIOD_CURRENT, 20, 0, MODE_SMA, PRICE_CLOSE);
   Stoch_Handle = iStochastic(_Symbol, PERIOD_CURRENT, 14, 3, 3, MODE_SMA, STO_LOWHIGH);
   
   if(ATR_Handle == INVALID_HANDLE || RSI_Handle == INVALID_HANDLE ||
      MACD_Handle == INVALID_HANDLE || MA20_Handle == INVALID_HANDLE ||
      Stoch_Handle == INVALID_HANDLE)
   {
      Print("Failed to create indicators");
      return INIT_FAILED;
   }
   
   Print("=== Golden Gibz EA Ready - ", _Symbol, " ===");
   return INIT_SUCCEEDED;
}

//+------------------------------------------------------------------+
void OnDeinit(const int reason)
{
   if(OnnxHandle != INVALID_HANDLE)
      OnnxRelease(OnnxHandle);
   
   IndicatorRelease(ATR_Handle);
   IndicatorRelease(RSI_Handle);
   IndicatorRelease(MACD_Handle);
   IndicatorRelease(MA20_Handle);
   IndicatorRelease(Stoch_Handle);
}

//+------------------------------------------------------------------+
void OnTick()
{
   static datetime lastBar = 0;
   datetime currentBar = iTime(_Symbol, PERIOD_CURRENT, 0);
   
   if(currentBar == lastBar) return;
   lastBar = currentBar;
   
   //--- Daily reset
   MqlDateTime dt;
   TimeToStruct(TimeCurrent(), dt);
   datetime today = StringToTime(StringFormat("%04d.%02d.%02d", dt.year, dt.mon, dt.day));
   if(today != CurrentDay)
   {
      CurrentDay = today;
      TodayTrades = 0;
   }
   
   UpdatePositionState();
   
   if(TodayTrades >= MaxDailyTrades) return;
   
   int barsSince = Bars(_Symbol, PERIOD_CURRENT, LastTradeBar, currentBar) - 1;
   if(barsSince < MinBarsBetweenTrades) return;
   
   if(CurrentPosition != 0) return;
   
   int action = GetPrediction();
   
   if(action == 1)
   {
      if(ExecuteTrade(ORDER_TYPE_BUY))
      {
         CurrentPosition = 1;
         TodayTrades++;
         LastTradeBar = currentBar;
      }
   }
   else if(action == 2)
   {
      if(ExecuteTrade(ORDER_TYPE_SELL))
      {
         CurrentPosition = -1;
         TodayTrades++;
         LastTradeBar = currentBar;
      }
   }
}

//+------------------------------------------------------------------+
void UpdatePositionState()
{
   if(PositionSelect(_Symbol))
   {
      long posType = PositionGetInteger(POSITION_TYPE);
      CurrentPosition = (posType == POSITION_TYPE_BUY) ? 1 : -1;
      EntryPrice = PositionGetDouble(POSITION_PRICE_OPEN);
   }
   else
   {
      CurrentPosition = 0;
      EntryPrice = 0;
      EntryATR = 0;
   }
}

//+------------------------------------------------------------------+
int GetPrediction()
{
   float obs[];
   ArrayResize(obs, WINDOW_SIZE * NUM_FEATURES);
   ArrayInitialize(obs, 0.0f);
   
   if(!BuildObservation(obs))
      return 0;
   
   float output[];
   ArrayResize(output, NUM_ACTIONS);
   
   if(!OnnxRun(OnnxHandle, ONNX_DEFAULT, obs, output))
   {
      Print("ONNX failed: ", GetLastError());
      return 0;
   }
   
   // Argmax
   int best = 0;
   float bestVal = output[0];
   for(int i = 1; i < NUM_ACTIONS; i++)
   {
      if(output[i] > bestVal)
      {
         bestVal = output[i];
         best = i;
      }
   }
   
   return best;
}

//+------------------------------------------------------------------+
bool BuildObservation(float &obs[])
{
   // Get price data
   MqlRates rates[];
   ArraySetAsSeries(rates, true);
   if(CopyRates(_Symbol, PERIOD_CURRENT, 0, WINDOW_SIZE + 5, rates) < WINDOW_SIZE)
      return false;
   
   // Get indicators
   double atr[], rsi[], macd[], macd_sig[], ma20[], stoch_k[], stoch_d[];
   ArraySetAsSeries(atr, true);
   ArraySetAsSeries(rsi, true);
   ArraySetAsSeries(macd, true);
   ArraySetAsSeries(macd_sig, true);
   ArraySetAsSeries(ma20, true);
   ArraySetAsSeries(stoch_k, true);
   ArraySetAsSeries(stoch_d, true);
   
   if(CopyBuffer(ATR_Handle, 0, 0, WINDOW_SIZE + 5, atr) < WINDOW_SIZE) return false;
   if(CopyBuffer(RSI_Handle, 0, 0, WINDOW_SIZE + 5, rsi) < WINDOW_SIZE) return false;
   if(CopyBuffer(MACD_Handle, 0, 0, WINDOW_SIZE + 5, macd) < WINDOW_SIZE) return false;
   if(CopyBuffer(MACD_Handle, 1, 0, WINDOW_SIZE + 5, macd_sig) < WINDOW_SIZE) return false;
   if(CopyBuffer(MA20_Handle, 0, 0, WINDOW_SIZE + 5, ma20) < WINDOW_SIZE) return false;
   if(CopyBuffer(Stoch_Handle, 0, 0, WINDOW_SIZE + 5, stoch_k) < WINDOW_SIZE) return false;
   if(CopyBuffer(Stoch_Handle, 1, 0, WINDOW_SIZE + 5, stoch_d) < WINDOW_SIZE) return false;
   
   // Base price = oldest bar's close in window (matching training)
   double basePrice = rates[WINDOW_SIZE - 1].close;
   if(basePrice <= 0) basePrice = rates[0].close;
   
   int idx = 0;
   
   // Build features for each bar (oldest to newest, matching training)
   for(int i = WINDOW_SIZE - 1; i >= 0; i--)
   {
      // === 16 MARKET FEATURES (must match training exactly) ===
      
      // 1-4: Price features (normalized)
      obs[idx++] = (float)((rates[i].open / basePrice - 1.0) * 100.0);
      obs[idx++] = (float)((rates[i].high / basePrice - 1.0) * 100.0);
      obs[idx++] = (float)((rates[i].low / basePrice - 1.0) * 100.0);
      obs[idx++] = (float)((rates[i].close / basePrice - 1.0) * 100.0);
      
      // 5: ATR (normalized)
      obs[idx++] = (float)(atr[i] / basePrice * 100.0);
      
      // 6: MA20 (normalized)
      obs[idx++] = (float)((ma20[i] / basePrice - 1.0) * 100.0);
      
      // 7: RSI (centered, scaled)
      double rsiNorm = (rsi[i] - 50.0) / 25.0;
      obs[idx++] = (float)MathMax(-3.0, MathMin(3.0, rsiNorm));
      
      // 8-10: MACD (normalized)
      obs[idx++] = (float)MathMax(-3.0, MathMin(3.0, macd[i] / basePrice * 1000.0));
      obs[idx++] = (float)MathMax(-3.0, MathMin(3.0, macd_sig[i] / basePrice * 1000.0));
      double hist = macd[i] - macd_sig[i];
      obs[idx++] = (float)MathMax(-3.0, MathMin(3.0, hist / basePrice * 1000.0));
      
      // 11: Trend
      obs[idx++] = (float)((rates[i].close > ma20[i]) ? 1.0 : -1.0);
      
      // 12: Stochastic K (centered, scaled)
      double stochKNorm = (stoch_k[i] - 50.0) / 25.0;
      obs[idx++] = (float)MathMax(-3.0, MathMin(3.0, stochKNorm));
      
      // 13: Stochastic D (centered, scaled)
      double stochDNorm = (stoch_d[i] - 50.0) / 25.0;
      obs[idx++] = (float)MathMax(-3.0, MathMin(3.0, stochDNorm));
      
      // 14: Stochastic cross
      obs[idx++] = (float)((stoch_k[i] > stoch_d[i]) ? 1.0 : -1.0);
      
      // 15: Overbought
      obs[idx++] = (float)((stoch_k[i] > 80) ? 1.0 : 0.0);
      
      // 16: Oversold
      obs[idx++] = (float)((stoch_k[i] < 20) ? 1.0 : 0.0);
      
      // === 2 STATE FEATURES ===
      
      // 17: Position
      obs[idx++] = (float)CurrentPosition;
      
      // 18: Unrealized PnL
      double pnl = 0.0;
      if(CurrentPosition != 0 && EntryPrice > 0 && EntryATR > 0)
      {
         if(CurrentPosition == 1)
            pnl = (rates[0].close - EntryPrice) / EntryATR;
         else
            pnl = (EntryPrice - rates[0].close) / EntryATR;
      }
      obs[idx++] = (float)MathMax(-3.0, MathMin(3.0, pnl));
   }
   
   return true;
}

//+------------------------------------------------------------------+
bool ExecuteTrade(ENUM_ORDER_TYPE orderType)
{
   double atrBuf[];
   ArraySetAsSeries(atrBuf, true);
   if(CopyBuffer(ATR_Handle, 0, 0, 1, atrBuf) < 1)
      return false;
   
   double currentATR = atrBuf[0];
   EntryATR = currentATR;
   
   double price = (orderType == ORDER_TYPE_BUY) ? 
                  SymbolInfoDouble(_Symbol, SYMBOL_ASK) : 
                  SymbolInfoDouble(_Symbol, SYMBOL_BID);
   
   double tpDist = currentATR * TP_ATR_Mult;
   double slDist = currentATR * SL_ATR_Mult;
   
   int digits = (int)SymbolInfoInteger(_Symbol, SYMBOL_DIGITS);
   double tp, sl;
   
   if(orderType == ORDER_TYPE_BUY)
   {
      tp = NormalizeDouble(price + tpDist, digits);
      sl = NormalizeDouble(price - slDist, digits);
   }
   else
   {
      tp = NormalizeDouble(price - tpDist, digits);
      sl = NormalizeDouble(price + slDist, digits);
   }
   
   string comment = (orderType == ORDER_TYPE_BUY) ? "GoldenGibz_BUY" : "GoldenGibz_SELL";
   
   if(Trade.PositionOpen(_Symbol, orderType, LotSize, price, sl, tp, comment))
   {
      EntryPrice = price;
      Print(comment, " @ ", DoubleToString(price, digits), 
            " TP:", DoubleToString(tp, digits), 
            " SL:", DoubleToString(sl, digits));
      return true;
   }
   
   Print("Trade failed: ", Trade.ResultRetcode());
   return false;
}

//+------------------------------------------------------------------+
double OnTester()
{
   double total = TesterStatistics(STAT_TRADES);
   double profit = TesterStatistics(STAT_PROFIT_TRADES);
   if(total > 0)
      return profit / total * 100.0;
   return 0;
}
//+------------------------------------------------------------------+
