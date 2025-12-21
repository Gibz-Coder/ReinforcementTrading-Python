//+------------------------------------------------------------------+
//|                                           HighWinRateEA_v2.mq5   |
//|                                    AI Trading Model Integration  |
//|                              Using embedded ONNX resource        |
//+------------------------------------------------------------------+
#property copyright "AI Trading System"
#property version   "2.00"
#property strict

#include <Trade\Trade.mqh>

//--- Embed ONNX model as resource
#resource "\\Files\\trading_model.onnx" as uchar OnnxModel[]

//--- Input parameters
input group "=== Trading Settings ==="
input double   LotSize = 0.01;
input double   TP_ATR_Mult = 0.5;
input double   SL_ATR_Mult = 2.0;
input int      MagicNumber = 888888;

input group "=== Risk Management ==="
input int      MaxDailyTrades = 20;
input int      MinBarsBetweenTrades = 3;

//--- Constants
#define WINDOW_SIZE     20
#define NUM_FEATURES    39
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
   Print("=== HighWinRate EA v2 Initializing ===");
   
   Trade.SetExpertMagicNumber(MagicNumber);
   Trade.SetDeviationInPoints(30);
   Trade.SetTypeFilling(ORDER_FILLING_IOC);
   
   //--- Load ONNX model from embedded resource
   ResetLastError();
   OnnxHandle = OnnxCreateFromBuffer(OnnxModel, ONNX_DEFAULT);
   
   if(OnnxHandle == INVALID_HANDLE)
   {
      int err = GetLastError();
      Print("Failed to load ONNX from resource. Error: ", err);
      Print("Resource size: ", ArraySize(OnnxModel), " bytes");
      return INIT_FAILED;
   }
   Print("ONNX loaded from embedded resource. Size: ", ArraySize(OnnxModel), " bytes");
   
   //--- Set input shape [1, 780]
   long input_shape[] = {1, WINDOW_SIZE * NUM_FEATURES};
   if(!OnnxSetInputShape(OnnxHandle, 0, input_shape))
   {
      Print("Failed to set input shape. Error: ", GetLastError());
      return INIT_FAILED;
   }
   
   //--- Set output shape [1, 3]
   long output_shape[] = {1, NUM_ACTIONS};
   if(!OnnxSetOutputShape(OnnxHandle, 0, output_shape))
   {
      Print("Failed to set output shape. Error: ", GetLastError());
      return INIT_FAILED;
   }
   Print("Shapes: input[1,", WINDOW_SIZE * NUM_FEATURES, "] output[1,", NUM_ACTIONS, "]");
   
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
   
   Print("=== EA Ready - Symbol: ", _Symbol, " ===");
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
   
   Print("EA stopped. Reason: ", reason);
}

//+------------------------------------------------------------------+
void OnTick()
{
   static datetime lastBar = 0;
   datetime currentBar = iTime(_Symbol, PERIOD_CURRENT, 0);
   
   if(currentBar == lastBar)
      return;
   lastBar = currentBar;
   
   //--- Reset daily counters
   MqlDateTime dt;
   TimeToStruct(TimeCurrent(), dt);
   datetime today = StringToTime(StringFormat("%04d.%02d.%02d", dt.year, dt.mon, dt.day));
   
   if(today != CurrentDay)
   {
      CurrentDay = today;
      TodayTrades = 0;
   }
   
   //--- Update position state
   UpdatePositionState();
   
   //--- Check limits
   if(TodayTrades >= MaxDailyTrades)
      return;
   
   int barsSince = Bars(_Symbol, PERIOD_CURRENT, LastTradeBar, currentBar) - 1;
   if(barsSince < MinBarsBetweenTrades)
      return;
   
   if(CurrentPosition != 0)
      return;
   
   //--- Get prediction
   int action = GetPrediction();
   
   //--- Execute
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
   static int callCount = 0;
   callCount++;
   
   float obs[];
   ArrayResize(obs, WINDOW_SIZE * NUM_FEATURES);
   ArrayInitialize(obs, 0.0f);
   
   if(!BuildObservation(obs))
      return 0;
   
   float output[];
   ArrayResize(output, NUM_ACTIONS);
   
   if(!OnnxRun(OnnxHandle, ONNX_DEFAULT, obs, output))
   {
      if(callCount <= 5) Print("ONNX run failed: ", GetLastError());
      return 0;
   }
   
   //--- Argmax
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
   
   // Debug first 10 predictions
   if(callCount <= 10)
   {
      Print("Bar ", callCount, " | H=", output[0], " B=", output[1], " S=", output[2], 
            " -> ", (best==0?"HOLD":(best==1?"BUY":"SELL")));
   }
   
   return best;
}

//+------------------------------------------------------------------+
bool BuildObservation(float &obs[])
{
   MqlRates rates[];
   ArraySetAsSeries(rates, true);
   if(CopyRates(_Symbol, PERIOD_CURRENT, 0, WINDOW_SIZE + 5, rates) < WINDOW_SIZE)
      return false;
   
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
   
   double basePrice = rates[WINDOW_SIZE - 1].close;
   if(basePrice <= 0) basePrice = rates[0].close;
   
   int idx = 0;
   
   for(int i = WINDOW_SIZE - 1; i >= 0; i--)
   {
      // Price features (4)
      obs[idx++] = (float)((rates[i].open / basePrice - 1.0) * 100.0);
      obs[idx++] = (float)((rates[i].high / basePrice - 1.0) * 100.0);
      obs[idx++] = (float)((rates[i].low / basePrice - 1.0) * 100.0);
      obs[idx++] = (float)((rates[i].close / basePrice - 1.0) * 100.0);
      
      // ATR (1)
      obs[idx++] = (float)(atr[i] / basePrice * 100.0);
      
      // MA20 (1)
      obs[idx++] = (float)((ma20[i] / basePrice - 1.0) * 100.0);
      
      // RSI (1)
      double rsiNorm = (rsi[i] - 50.0) / 25.0;
      obs[idx++] = (float)MathMax(-3.0, MathMin(3.0, rsiNorm));
      
      // MACD (3)
      obs[idx++] = (float)MathMax(-3.0, MathMin(3.0, macd[i] / basePrice * 1000.0));
      obs[idx++] = (float)MathMax(-3.0, MathMin(3.0, macd_sig[i] / basePrice * 1000.0));
      double hist = macd[i] - macd_sig[i];
      obs[idx++] = (float)MathMax(-3.0, MathMin(3.0, hist / basePrice * 1000.0));
      
      // Trend (1)
      obs[idx++] = (float)((rates[i].close > ma20[i]) ? 1.0 : -1.0);
      
      // Stochastic features (12)
      double stochNorm = (stoch_k[i] - 50.0) / 25.0;
      obs[idx++] = (float)MathMax(-3.0, MathMin(3.0, stochNorm));
      obs[idx++] = (float)((stoch_k[i] > stoch_d[i]) ? -1.0 : 1.0);
      obs[idx++] = (float)((stoch_k[i] < 20 && stoch_k[i] > stoch_d[i]) ? 1.0 : 0.0);
      obs[idx++] = (float)((stoch_k[i] > 80 && stoch_k[i] < stoch_d[i]) ? 1.0 : 0.0);
      obs[idx++] = (float)((stoch_k[i] > 80) ? 1.0 : 0.0);
      obs[idx++] = (float)((stoch_k[i] < 20) ? 1.0 : 0.0);
      obs[idx++] = (float)((stoch_k[i] > 90) ? 1.0 : 0.0);
      obs[idx++] = (float)((stoch_k[i] < 10) ? 1.0 : 0.0);
      obs[idx++] = (float)(MathAbs(stoch_k[i] - 50.0) / 50.0);
      double mom = (i < WINDOW_SIZE - 1) ? (stoch_k[i] - stoch_k[i+1]) / 10.0 : 0.0;
      obs[idx++] = (float)MathMax(-3.0, MathMin(3.0, mom));
      obs[idx++] = (float)((stoch_k[i] > stoch_d[i]) ? -1.0 : 1.0);
      obs[idx++] = (float)(MathAbs(stoch_k[i] - 50.0) / 50.0);
      
      // Padding (4)
      obs[idx++] = 0.0f;
      obs[idx++] = 0.0f;
      obs[idx++] = 0.0f;
      obs[idx++] = 0.0f;
      
      // Volume (1)
      obs[idx++] = 0.0f;
      
      // State features (3)
      obs[idx++] = (float)CurrentPosition;
      double pnl = 0.0;
      if(CurrentPosition != 0 && EntryPrice > 0 && EntryATR > 0)
      {
         if(CurrentPosition == 1)
            pnl = (rates[0].close - EntryPrice) / EntryATR;
         else
            pnl = (EntryPrice - rates[0].close) / EntryATR;
      }
      obs[idx++] = (float)MathMax(-3.0, MathMin(3.0, pnl));
      obs[idx++] = 0.0f;
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
   
   string comment = (orderType == ORDER_TYPE_BUY) ? "AI_BUY" : "AI_SELL";
   
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
