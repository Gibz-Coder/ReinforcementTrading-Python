//+------------------------------------------------------------------+
//|                                              GoldenGibzEA.mq5    |
//|                                  Golden-Gibz Hybrid System v1.0  |
//|                                   Reads signals from Python ML   |
//+------------------------------------------------------------------+
#property copyright "Golden-Gibz"
#property version   "1.00"
#property description "Executes trades based on Golden-Gibz ML signals"

#include <Trade\Trade.mqh>
#include <Files\FileTxt.mqh>

//--- Input parameters
input group "=== SIGNAL SETTINGS ==="
input string SignalFile = "signals.json";           // Signal file name
input int SignalTimeoutSec = 1800;                  // Signal timeout (30 min)
input bool EnableTrading = true;                    // Enable actual trading

input group "=== RISK MANAGEMENT ==="
input double LotSize = 0.01;                       // Fixed lot size
input double MaxRiskPercent = 2.0;                 // Max risk per trade (%)
input double StopLossATRMultiplier = 2.0;          // Stop loss ATR multiplier
input double TakeProfitATRMultiplier = 2.0;        // Take profit ATR multiplier
input int MaxPositions = 1;                        // Max concurrent positions

input group "=== TRADING HOURS ==="
input int StartHour = 8;                           // Trading start hour
input int EndHour = 17;                            // Trading end hour
input bool TradeOnFriday = false;                  // Trade on Friday

input group "=== SAFETY ==="
input double MaxDailyLoss = 100.0;                // Max daily loss ($)
input int MaxDailyTrades = 10;                     // Max trades per day
input double MinConfidence = 0.6;                  // Minimum signal confidence

//--- Global variables
CTrade trade;
CFileTxt signalFile;
datetime lastSignalTime = 0;
datetime lastTradeTime = 0;
double dailyPnL = 0.0;
int dailyTrades = 0;
datetime currentDay = 0;

//--- Signal structure
struct SignalData
{
   datetime timestamp;
   int action;                    // 0=HOLD, 1=LONG, 2=SHORT
   string actionName;
   double confidence;
   double price;
   int bullTimeframes;
   int bearTimeframes;
   int trendStrength;
   double rsi;
   double atrPct;
   bool bullSignal;
   bool bearSignal;
   bool bullPullback;
   bool bearPullback;
   bool activeSession;
   double atrValue;
   double stopDistance;
   double targetDistance;
};

//+------------------------------------------------------------------+
//| Expert initialization function                                   |
//+------------------------------------------------------------------+
int OnInit()
{
   Print("🎯 Golden-Gibz EA Starting...");
   
   // Initialize trade object
   trade.SetExpertMagicNumber(12345);
   trade.SetDeviationInPoints(10);
   trade.SetTypeFilling(ORDER_FILLING_IOC);
   
   // Reset daily counters
   ResetDailyCounters();
   
   Print("✅ Golden-Gibz EA Initialized");
   Print("   Signal File: ", SignalFile);
   Print("   Lot Size: ", LotSize);
   Print("   Max Risk: ", MaxRiskPercent, "%");
   Print("   Trading Hours: ", StartHour, ":00 - ", EndHour, ":00");
   
   return INIT_SUCCEEDED;
}

//+------------------------------------------------------------------+
//| Expert deinitialization function                                |
//+------------------------------------------------------------------+
void OnDeinit(const int reason)
{
   Print("🛑 Golden-Gibz EA Stopped");
   signalFile.Close();
}

//+------------------------------------------------------------------+
//| Expert tick function                                             |
//+------------------------------------------------------------------+
void OnTick()
{
   // Check if new day
   MqlDateTime currentTime, dayTime;
   TimeToStruct(TimeCurrent(), currentTime);
   
   // Initialize currentDay if it's zero (first run)
   if (currentDay == 0)
   {
      ResetDailyCounters();
      return;
   }
   
   TimeToStruct(currentDay, dayTime);
   
   if (currentTime.day != dayTime.day)
   {
      ResetDailyCounters();
   }
   
   // Safety checks
   if (!EnableTrading)
      return;
      
   if (!IsTradingTime())
      return;
      
   if (dailyPnL <= -MaxDailyLoss)
   {
      Comment("❌ Daily loss limit reached: $", dailyPnL);
      return;
   }
   
   if (dailyTrades >= MaxDailyTrades)
   {
      Comment("❌ Daily trade limit reached: ", dailyTrades);
      return;
   }
   
   // Read and process signals
   SignalData signal;
   if (ReadSignal(signal))
   {
      ProcessSignal(signal);
   }
   
   // Update display
   UpdateDisplay();
}

//+------------------------------------------------------------------+
//| Read signal from JSON file                                      |
//+------------------------------------------------------------------+
bool ReadSignal(SignalData &signal)
{
   // Try to open signal file
   string fullPath = TerminalInfoString(TERMINAL_DATA_PATH) + "\\MQL5\\Files\\" + SignalFile;
   
   // Reset file handle
   signalFile.Close();
   
   if (!signalFile.Open(fullPath, FILE_READ|FILE_TXT))
   {
      // File not found or cannot be opened - this is normal when no signal exists yet
      return false;
   }
   
   // Read entire file content
   string jsonContent = "";
   while (!signalFile.IsEnding())
   {
      jsonContent += signalFile.ReadString() + "\n";
   }
   signalFile.Close();
   
   if (StringLen(jsonContent) < 10)
      return false;
   
   // Parse JSON (simplified parsing for key fields)
   if (!ParseSignalJSON(jsonContent, signal))
      return false;
   
   // Check if signal is fresh
   datetime signalTime = signal.timestamp;
   if (signalTime <= lastSignalTime)
      return false;  // Already processed
   
   if ((TimeCurrent() - signalTime) > SignalTimeoutSec)
   {
      Print("⚠️ Signal too old: ", (TimeCurrent() - signalTime), " seconds");
      return false;
   }
   
   lastSignalTime = signalTime;
   return true;
}

//+------------------------------------------------------------------+
//| Simple JSON parser for signal data                              |
//+------------------------------------------------------------------+
bool ParseSignalJSON(string json, SignalData &signal)
{
   // This is a simplified parser - in production, use a proper JSON library
   
   // Extract timestamp
   string timestampStr = ExtractJSONValue(json, "timestamp");
   signal.timestamp = StringToTime(StringSubstr(timestampStr, 0, 19)); // ISO format
   
   // Extract action
   signal.action = (int)StringToInteger(ExtractJSONValue(json, "action"));
   signal.actionName = ExtractJSONValue(json, "action_name");
   
   // Extract confidence
   signal.confidence = StringToDouble(ExtractJSONValue(json, "confidence"));
   
   // Extract market conditions
   signal.price = StringToDouble(ExtractJSONValue(json, "price"));
   signal.bullTimeframes = (int)StringToInteger(ExtractJSONValue(json, "bull_timeframes"));
   signal.bearTimeframes = (int)StringToInteger(ExtractJSONValue(json, "bear_timeframes"));
   signal.trendStrength = (int)StringToInteger(ExtractJSONValue(json, "trend_strength"));
   signal.rsi = StringToDouble(ExtractJSONValue(json, "rsi"));
   signal.atrPct = StringToDouble(ExtractJSONValue(json, "atr_pct"));
   
   // Extract boolean signals
   signal.bullSignal = (ExtractJSONValue(json, "bull_signal") == "true");
   signal.bearSignal = (ExtractJSONValue(json, "bear_signal") == "true");
   signal.bullPullback = (ExtractJSONValue(json, "bull_pullback") == "true");
   signal.bearPullback = (ExtractJSONValue(json, "bear_pullback") == "true");
   signal.activeSession = (ExtractJSONValue(json, "active_session") == "true");
   
   // Extract risk management
   signal.atrValue = StringToDouble(ExtractJSONValue(json, "atr_value"));
   signal.stopDistance = StringToDouble(ExtractJSONValue(json, "stop_distance"));
   signal.targetDistance = StringToDouble(ExtractJSONValue(json, "target_distance"));
   
   return true;
}

//+------------------------------------------------------------------+
//| Extract value from JSON string (simplified)                     |
//+------------------------------------------------------------------+
string ExtractJSONValue(string json, string key)
{
   string searchKey = "\"" + key + "\":";
   int pos = StringFind(json, searchKey);
   if (pos == -1)
      return "";
   
   pos += StringLen(searchKey);
   
   // Skip whitespace and quotes
   while (pos < StringLen(json) && (StringGetCharacter(json, pos) == ' ' || StringGetCharacter(json, pos) == '"'))
      pos++;
   
   int endPos = pos;
   bool inQuotes = false;
   
   // Find end of value
   while (endPos < StringLen(json))
   {
      ushort ch = StringGetCharacter(json, endPos);
      if (ch == '"')
         inQuotes = !inQuotes;
      else if (!inQuotes && (ch == ',' || ch == '}' || ch == '\n'))
         break;
      endPos++;
   }
   
   string value = StringSubstr(json, pos, endPos - pos);
   
   // Remove quotes if present
   if (StringLen(value) >= 2 && StringGetCharacter(value, 0) == '"')
      value = StringSubstr(value, 1, StringLen(value) - 2);
   
   return value;
}

//+------------------------------------------------------------------+
//| Process trading signal                                           |
//+------------------------------------------------------------------+
void ProcessSignal(SignalData &signal)
{
   Print("📊 New Signal: ", signal.actionName, " (Confidence: ", signal.confidence, ")");
   Print("   Market: Bull TF=", signal.bullTimeframes, ", Bear TF=", signal.bearTimeframes, 
         ", Strength=", signal.trendStrength);
   
   // Check confidence threshold
   if (signal.confidence < MinConfidence)
   {
      Print("⚠️ Signal confidence too low: ", signal.confidence, " < ", MinConfidence);
      return;
   }
   
   // Check if we already have a position
   if (PositionsTotal() >= MaxPositions)
   {
      Print("⚠️ Max positions reached: ", PositionsTotal());
      return;
   }
   
   // Cooldown between trades (prevent overtrading)
   if ((TimeCurrent() - lastTradeTime) < 300)  // 5 minutes
   {
      Print("⚠️ Trade cooldown active");
      return;
   }
   
   // Execute signal
   if (signal.action == 1)  // LONG
   {
      ExecuteLongTrade(signal);
   }
   else if (signal.action == 2)  // SHORT
   {
      ExecuteShortTrade(signal);
   }
   // action == 0 (HOLD) - do nothing
}

//+------------------------------------------------------------------+
//| Execute long trade                                               |
//+------------------------------------------------------------------+
void ExecuteLongTrade(SignalData &signal)
{
   double ask = SymbolInfoDouble(_Symbol, SYMBOL_ASK);
   double sl = ask - (signal.stopDistance * StopLossATRMultiplier);
   double tp = ask + (signal.targetDistance * TakeProfitATRMultiplier);
   
   // Adjust lot size based on risk
   double adjustedLots = CalculateRiskBasedLots(ask, sl);
   
   Print("🟢 Executing LONG trade:");
   Print("   Price: ", ask);
   Print("   SL: ", sl, " (Distance: ", (ask - sl), ")");
   Print("   TP: ", tp, " (Distance: ", (tp - ask), ")");
   Print("   Lots: ", adjustedLots);
   
   if (trade.Buy(adjustedLots, _Symbol, ask, sl, tp, "Golden-Gibz LONG"))
   {
      Print("✅ LONG trade executed successfully");
      lastTradeTime = TimeCurrent();
      dailyTrades++;
   }
   else
   {
      Print("❌ LONG trade failed: ", trade.ResultRetcode(), " - ", trade.ResultRetcodeDescription());
   }
}

//+------------------------------------------------------------------+
//| Execute short trade                                              |
//+------------------------------------------------------------------+
void ExecuteShortTrade(SignalData &signal)
{
   double bid = SymbolInfoDouble(_Symbol, SYMBOL_BID);
   double sl = bid + (signal.stopDistance * StopLossATRMultiplier);
   double tp = bid - (signal.targetDistance * TakeProfitATRMultiplier);
   
   // Adjust lot size based on risk
   double adjustedLots = CalculateRiskBasedLots(bid, sl);
   
   Print("🔴 Executing SHORT trade:");
   Print("   Price: ", bid);
   Print("   SL: ", sl, " (Distance: ", (sl - bid), ")");
   Print("   TP: ", tp, " (Distance: ", (bid - tp), ")");
   Print("   Lots: ", adjustedLots);
   
   if (trade.Sell(adjustedLots, _Symbol, bid, sl, tp, "Golden-Gibz SHORT"))
   {
      Print("✅ SHORT trade executed successfully");
      lastTradeTime = TimeCurrent();
      dailyTrades++;
   }
   else
   {
      Print("❌ SHORT trade failed: ", trade.ResultRetcode(), " - ", trade.ResultRetcodeDescription());
   }
}

//+------------------------------------------------------------------+
//| Calculate risk-based lot size                                    |
//+------------------------------------------------------------------+
double CalculateRiskBasedLots(double entryPrice, double stopLoss)
{
   double accountBalance = AccountInfoDouble(ACCOUNT_BALANCE);
   double riskAmount = accountBalance * (MaxRiskPercent / 100.0);
   
   double stopDistance = MathAbs(entryPrice - stopLoss);
   double tickValue = SymbolInfoDouble(_Symbol, SYMBOL_TRADE_TICK_VALUE);
   double tickSize = SymbolInfoDouble(_Symbol, SYMBOL_TRADE_TICK_SIZE);
   
   double riskPerLot = (stopDistance / tickSize) * tickValue;
   double calculatedLots = riskAmount / riskPerLot;
   
   // Apply limits
   double minLot = SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_MIN);
   double maxLot = SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_MAX);
   double lotStep = SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_STEP);
   
   calculatedLots = MathMax(calculatedLots, minLot);
   calculatedLots = MathMin(calculatedLots, maxLot);
   calculatedLots = MathMin(calculatedLots, LotSize * 3);  // Don't exceed 3x fixed lot size
   
   // Round to lot step
   calculatedLots = MathFloor(calculatedLots / lotStep) * lotStep;
   
   return MathMax(calculatedLots, minLot);
}

//+------------------------------------------------------------------+
//| Check if it's trading time                                       |
//+------------------------------------------------------------------+
bool IsTradingTime()
{
   MqlDateTime dt;
   TimeToStruct(TimeCurrent(), dt);
   
   // Check day of week
   if (dt.day_of_week == 0)  // Sunday
      return false;
   if (dt.day_of_week == 6)  // Saturday
      return false;
   if (dt.day_of_week == 5 && !TradeOnFriday)  // Friday
      return false;
   
   // Check hour
   if (dt.hour < StartHour || dt.hour >= EndHour)
      return false;
   
   return true;
}

//+------------------------------------------------------------------+
//| Reset daily counters                                             |
//+------------------------------------------------------------------+
void ResetDailyCounters()
{
   currentDay = TimeCurrent();
   dailyPnL = CalculateDailyPnL();
   dailyTrades = CountDailyTrades();
   
   Print("📅 Daily counters reset - PnL: $", dailyPnL, ", Trades: ", dailyTrades);
}

//+------------------------------------------------------------------+
//| Calculate daily P&L                                              |
//+------------------------------------------------------------------+
double CalculateDailyPnL()
{
   double pnl = 0.0;
   datetime dayStart = StringToTime(TimeToString(TimeCurrent(), TIME_DATE));
   
   // Check closed positions
   HistorySelect(dayStart, TimeCurrent());
   for (int i = 0; i < HistoryDealsTotal(); i++)
   {
      ulong ticket = HistoryDealGetTicket(i);
      if (HistoryDealGetString(ticket, DEAL_SYMBOL) == _Symbol)
      {
         pnl += HistoryDealGetDouble(ticket, DEAL_PROFIT);
      }
   }
   
   // Add open positions floating P&L
   for (int i = 0; i < PositionsTotal(); i++)
   {
      if (PositionGetSymbol(i) == _Symbol)
      {
         pnl += PositionGetDouble(POSITION_PROFIT);
      }
   }
   
   return pnl;
}

//+------------------------------------------------------------------+
//| Count daily trades                                               |
//+------------------------------------------------------------------+
int CountDailyTrades()
{
   int count = 0;
   datetime dayStart = StringToTime(TimeToString(TimeCurrent(), TIME_DATE));
   
   HistorySelect(dayStart, TimeCurrent());
   for (int i = 0; i < HistoryDealsTotal(); i++)
   {
      ulong ticket = HistoryDealGetTicket(i);
      if (HistoryDealGetString(ticket, DEAL_SYMBOL) == _Symbol &&
          HistoryDealGetInteger(ticket, DEAL_TYPE) <= 1)  // Buy or Sell
      {
         count++;
      }
   }
   
   return count;
}

//+------------------------------------------------------------------+
//| Update display information                                       |
//+------------------------------------------------------------------+
void UpdateDisplay()
{
   string info = "🎯 Golden-Gibz EA\n";
   info += "Status: " + (EnableTrading ? "ACTIVE" : "DISABLED") + "\n";
   info += "Trading Time: " + (IsTradingTime() ? "YES" : "NO") + "\n";
   info += "Positions: " + IntegerToString(PositionsTotal()) + "/" + IntegerToString(MaxPositions) + "\n";
   info += "Daily P&L: $" + DoubleToString(CalculateDailyPnL(), 2) + "\n";
   info += "Daily Trades: " + IntegerToString(dailyTrades) + "/" + IntegerToString(MaxDailyTrades) + "\n";
   info += "Last Signal: " + TimeToString(lastSignalTime) + "\n";
   
   Comment(info);
}