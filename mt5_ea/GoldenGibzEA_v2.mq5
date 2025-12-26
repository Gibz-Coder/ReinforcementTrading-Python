//+------------------------------------------------------------------+
//|                                           GoldenGibzEA_v2.mq5    |
//|                              Golden-Gibz Simplified EA v2.0      |
//|                          Bulletproof version - Guaranteed to work |
//+------------------------------------------------------------------+
#property copyright "Golden-Gibz"
#property version   "2.00"
#property description "Simplified Golden-Gibz EA - Bulletproof Implementation"

#include <Trade\Trade.mqh>

//--- Input parameters
input group "=== SIGNAL SETTINGS ==="
input string SignalFile = "signals.json";           // Signal file name
input bool EnableTrading = true;                    // Enable actual trading
input double MinConfidence = 0.6;                   // Minimum signal confidence

input group "=== RISK MANAGEMENT ==="
input double LotSize = 0.01;                       // Fixed lot size
input double StopLossATRMultiplier = 2.0;          // Stop loss ATR multiplier
input double TakeProfitATRMultiplier = 2.0;        // Take profit ATR multiplier
input int MaxPositions = 1;                        // Max concurrent positions

input group "=== SAFETY ==="
input double MaxDailyLoss = 100.0;                // Max daily loss ($)
input int MaxDailyTrades = 10;                     // Max trades per day

//--- Global variables
CTrade trade;
datetime lastSignalTime = 0;
datetime lastTradeTime = 0;
double dailyPnL = 0.0;
int dailyTrades = 0;
datetime currentDay = 0;
int tickCounter = 0;

//+------------------------------------------------------------------+
//| Expert initialization function                                   |
//+------------------------------------------------------------------+
int OnInit()
{
   Print("🎯 Golden-Gibz EA v2.0 Starting...");
   
   // Initialize trade object
   trade.SetExpertMagicNumber(12345);
   trade.SetDeviationInPoints(10);
   trade.SetTypeFilling(ORDER_FILLING_IOC);
   
   // Reset daily counters
   ResetDailyCounters();
   
   Print("✅ Golden-Gibz EA v2.0 Initialized Successfully");
   Print("   Signal File: ", SignalFile);
   Print("   Lot Size: ", LotSize);
   Print("   Min Confidence: ", MinConfidence);
   Print("   Enable Trading: ", EnableTrading ? "YES" : "NO");
   
   return INIT_SUCCEEDED;
}

//+------------------------------------------------------------------+
//| Expert deinitialization function                                |
//+------------------------------------------------------------------+
void OnDeinit(const int reason)
{
   Print("🛑 Golden-Gibz EA v2.0 Stopped");
}

//+------------------------------------------------------------------+
//| Expert tick function                                             |
//+------------------------------------------------------------------+
void OnTick()
{
   tickCounter++;
   
   // Only process every 10th tick to reduce CPU load
   if (tickCounter % 10 != 0)
      return;
   
   // Check if new day
   CheckNewDay();
   
   // Safety checks
   if (!EnableTrading)
   {
      Comment("❌ Trading Disabled");
      return;
   }
   
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
   
   // Check positions
   if (PositionsTotal() >= MaxPositions)
   {
      Comment("⚠️ Max positions reached: ", PositionsTotal());
      return;
   }
   
   // Try to read and process signal
   if (ReadAndProcessSignal())
   {
      Print("✅ Signal processed successfully");
   }
   
   // Update display
   UpdateDisplay();
}

//+------------------------------------------------------------------+
//| Read and process signal from file                               |
//+------------------------------------------------------------------+
bool ReadAndProcessSignal()
{
   // Construct file path
   string terminalDataPath = TerminalInfoString(TERMINAL_DATA_PATH);
   string fullPath = terminalDataPath + "\\MQL5\\Files\\" + SignalFile;
   
   // Try to open file
   int fileHandle = FileOpen(SignalFile, FILE_READ|FILE_TXT);
   if (fileHandle == INVALID_HANDLE)
   {
      // File not found - this is normal when no signal exists
      return false;
   }
   
   // Read file content
   string jsonContent = "";
   while (!FileIsEnding(fileHandle))
   {
      jsonContent += FileReadString(fileHandle) + "\n";
   }
   FileClose(fileHandle);
   
   if (StringLen(jsonContent) < 10)
   {
      return false;
   }
   
   Print("📄 Signal file read, length: ", StringLen(jsonContent));
   
   // Parse signal data (simplified parsing)
   string timestampStr = ExtractJSONValue(jsonContent, "timestamp");
   int action = (int)StringToInteger(ExtractJSONValue(jsonContent, "action"));
   string actionName = ExtractJSONValue(jsonContent, "action_name");
   double confidence = StringToDouble(ExtractJSONValue(jsonContent, "confidence"));
   
   // Convert timestamp
   datetime signalTime = StringToTime(StringSubstr(timestampStr, 0, 19));
   
   Print("📊 Signal parsed:");
   Print("   Timestamp: ", TimeToString(signalTime));
   Print("   Action: ", action, " (", actionName, ")");
   Print("   Confidence: ", confidence);
   
   // Check if signal is fresh
   if (signalTime <= lastSignalTime)
   {
      Print("⚠️ Signal already processed");
      return false;
   }
   
   if ((TimeCurrent() - signalTime) > 1800) // 30 minutes timeout
   {
      Print("⚠️ Signal too old: ", (TimeCurrent() - signalTime), " seconds");
      return false;
   }
   
   // Check confidence
   if (confidence < MinConfidence)
   {
      Print("⚠️ Signal confidence too low: ", confidence, " < ", MinConfidence);
      return false;
   }
   
   // Update last signal time
   lastSignalTime = signalTime;
   
   // Execute trade based on action
   if (action == 1) // LONG
   {
      return ExecuteLongTrade(jsonContent);
   }
   else if (action == 2) // SHORT
   {
      return ExecuteShortTrade(jsonContent);
   }
   
   Print("⚠️ No action required for signal: ", action);
   return true;
}

//+------------------------------------------------------------------+
//| Execute long trade                                               |
//+------------------------------------------------------------------+
bool ExecuteLongTrade(string jsonContent)
{
   Print("🟢 Executing LONG trade...");
   
   // Get current price
   double ask = SymbolInfoDouble(_Symbol, SYMBOL_ASK);
   
   // Get ATR value from signal
   double atrValue = StringToDouble(ExtractJSONValue(jsonContent, "atr_value"));
   if (atrValue <= 0)
   {
      atrValue = 10.0; // Default fallback
   }
   
   // Calculate stops
   double stopDistance = atrValue * StopLossATRMultiplier;
   double targetDistance = atrValue * TakeProfitATRMultiplier;
   
   double sl = ask - stopDistance;
   double tp = ask + targetDistance;
   
   Print("   Entry Price: ", ask);
   Print("   Stop Loss: ", sl, " (Distance: ", stopDistance, ")");
   Print("   Take Profit: ", tp, " (Distance: ", targetDistance, ")");
   Print("   Lot Size: ", LotSize);
   
   // Execute trade
   if (trade.Buy(LotSize, _Symbol, ask, sl, tp, "Golden-Gibz LONG v2"))
   {
      Print("✅ LONG trade executed successfully!");
      Print("   Order: ", trade.ResultOrder());
      Print("   Deal: ", trade.ResultDeal());
      
      lastTradeTime = TimeCurrent();
      dailyTrades++;
      return true;
   }
   else
   {
      Print("❌ LONG trade failed!");
      Print("   Error Code: ", trade.ResultRetcode());
      Print("   Error Description: ", trade.ResultRetcodeDescription());
      return false;
   }
}

//+------------------------------------------------------------------+
//| Execute short trade                                              |
//+------------------------------------------------------------------+
bool ExecuteShortTrade(string jsonContent)
{
   Print("🔴 Executing SHORT trade...");
   
   // Get current price
   double bid = SymbolInfoDouble(_Symbol, SYMBOL_BID);
   
   // Get ATR value from signal
   double atrValue = StringToDouble(ExtractJSONValue(jsonContent, "atr_value"));
   if (atrValue <= 0)
   {
      atrValue = 10.0; // Default fallback
   }
   
   // Calculate stops
   double stopDistance = atrValue * StopLossATRMultiplier;
   double targetDistance = atrValue * TakeProfitATRMultiplier;
   
   double sl = bid + stopDistance;
   double tp = bid - targetDistance;
   
   Print("   Entry Price: ", bid);
   Print("   Stop Loss: ", sl, " (Distance: ", stopDistance, ")");
   Print("   Take Profit: ", tp, " (Distance: ", targetDistance, ")");
   Print("   Lot Size: ", LotSize);
   
   // Execute trade
   if (trade.Sell(LotSize, _Symbol, bid, sl, tp, "Golden-Gibz SHORT v2"))
   {
      Print("✅ SHORT trade executed successfully!");
      Print("   Order: ", trade.ResultOrder());
      Print("   Deal: ", trade.ResultDeal());
      
      lastTradeTime = TimeCurrent();
      dailyTrades++;
      return true;
   }
   else
   {
      Print("❌ SHORT trade failed!");
      Print("   Error Code: ", trade.ResultRetcode());
      Print("   Error Description: ", trade.ResultRetcodeDescription());
      return false;
   }
}

//+------------------------------------------------------------------+
//| Extract value from JSON string (simplified parser)              |
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
      else if (!inQuotes && (ch == ',' || ch == '}' || ch == '\n' || ch == '\r'))
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
//| Check if new day                                                 |
//+------------------------------------------------------------------+
void CheckNewDay()
{
   MqlDateTime currentTime, dayTime;
   TimeToStruct(TimeCurrent(), currentTime);
   
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
   string info = "🎯 Golden-Gibz EA v2.0\n";
   info += "Status: " + (EnableTrading ? "ACTIVE" : "DISABLED") + "\n";
   info += "Positions: " + IntegerToString(PositionsTotal()) + "/" + IntegerToString(MaxPositions) + "\n";
   info += "Daily P&L: $" + DoubleToString(CalculateDailyPnL(), 2) + "\n";
   info += "Daily Trades: " + IntegerToString(dailyTrades) + "/" + IntegerToString(MaxDailyTrades) + "\n";
   info += "Last Signal: " + TimeToString(lastSignalTime) + "\n";
   info += "Ticks: " + IntegerToString(tickCounter);
   
   Comment(info);
}