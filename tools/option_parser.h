/*
 * Copyright (C) 2025 Georg Wieser
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */

#ifndef OPTION_PARSER_H
#define OPTION_PARSER_H

/**
 * @file option_parser.h
 * @brief Lightweight command-line option parser replacing boost::program_options.
 *
 * Supports bool switches, string/int/uint values, short aliases,
 * case-insensitive matching, default and implicit values, and
 * per-option notifier callbacks.
 */

#include <string>
#include <vector>
#include <functional>
#include <iostream>
#include <sstream>
#include <algorithm>
#include <cctype>
#include <stdexcept>

class OptionDesc
{
public:
	explicit OptionDesc(const std::string& title = "")
		: m_title(title) {}

	/**
	 * Add a boolean switch.  --flag / -f  (no argument).
	 * @param nameSpec  "longname" or "longname,c" where c is the short alias.
	 */
	void addBoolSwitch(const std::string& nameSpec,
	                   std::function<void(bool)> cb,
	                   const std::string& desc)
	{
		Option o;
		parseName(nameSpec, o.longName, o.shortName);
		o.type = BOOL_SWITCH;
		o.description = desc;
		o.boolCb = std::move(cb);
		m_options.push_back(std::move(o));
	}

	/**
	 * Add a string-valued option.  --key=val / --key val / -k val
	 */
	void addStringOption(const std::string& nameSpec,
	                     const std::string& defaultVal,
	                     std::function<void(const std::string&)> cb,
	                     const std::string& desc)
	{
		Option o;
		parseName(nameSpec, o.longName, o.shortName);
		o.type = STRING_VALUE;
		o.defaultVal = defaultVal;
		o.description = desc;
		o.stringCb = std::move(cb);
		m_options.push_back(std::move(o));
	}

	/**
	 * Add an integer-valued option.  --key=N / --key N / -k N
	 */
	void addIntOption(const std::string& nameSpec,
	                  int defaultVal,
	                  std::function<void(int)> cb,
	                  const std::string& desc)
	{
		Option o;
		parseName(nameSpec, o.longName, o.shortName);
		o.type = INT_VALUE;
		o.defaultVal = std::to_string(defaultVal);
		o.description = desc;
		o.intCb = std::move(cb);
		m_options.push_back(std::move(o));
	}

	/**
	 * Add an unsigned-int option with an optional implicit value.
	 * E.g. -v means --verbose=1 (implicit), -v 3 means --verbose=3.
	 * Set implicitVal < 0 to disable implicit-value behaviour.
	 */
	void addUintOption(const std::string& nameSpec,
	                   unsigned int defaultVal,
	                   int implicitVal,
	                   std::function<void(unsigned int)> cb,
	                   const std::string& desc)
	{
		Option o;
		parseName(nameSpec, o.longName, o.shortName);
		o.type = UINT_VALUE;
		o.defaultVal = std::to_string(defaultVal);
		o.hasImplicit = (implicitVal >= 0);
		o.implicitVal = std::to_string(implicitVal >= 0 ? implicitVal : 0);
		o.description = desc;
		o.uintCb = std::move(cb);
		m_options.push_back(std::move(o));
	}

	/** Merge another OptionDesc into this one. */
	void merge(const OptionDesc& other)
	{
		m_options.insert(m_options.end(),
		                 other.m_options.begin(),
		                 other.m_options.end());
	}

	/**
	 * Parse command-line arguments.
	 * Non-option positional arguments (not starting with '-') are silently
	 * skipped, allowing e.g. `openEMS sim.xml --engine=basic`.
	 * After parsing, notifier callbacks are fired (defaults first, then
	 * explicitly-set values).
	 */
	void parse(int argc, const char* argv[]) const
	{
		std::vector<std::string> args;
		for (int i = 1; i < argc; ++i)  // skip program name
			args.push_back(argv[i]);
		parse(args);
	}

	/**
	 * Parse a vector of argument strings (library / Python binding mode).
	 * Each string is expected to already have a leading '--' or '-' prefix.
	 */
	void parse(const std::vector<std::string>& args) const
	{
		// Which options were explicitly set?
		std::vector<bool> wasSet(m_options.size(), false);
		std::vector<std::string> values(m_options.size());

		// Initialise with defaults
		for (size_t i = 0; i < m_options.size(); ++i)
			values[i] = m_options[i].defaultVal;

		for (size_t a = 0; a < args.size(); ++a)
		{
			const std::string& arg = args[a];

			if (arg.empty() || arg[0] != '-')
				continue;   // skip positional args

			// Split "--key=val" into key and val
			std::string key;
			std::string val;
			bool hasEquals = false;

			if (arg.size() > 2 && arg[0] == '-' && arg[1] == '-')
			{
				// Long option
				auto eq = arg.find('=', 2);
				if (eq != std::string::npos)
				{
					key = arg.substr(2, eq - 2);
					val = arg.substr(eq + 1);
					hasEquals = true;
				}
				else
				{
					key = arg.substr(2);
				}
			}
			else if (arg.size() == 2 && arg[0] == '-')
			{
				// Short option  -x
				key = arg.substr(1);
			}
			else
			{
				continue; // e.g. bare "-" or something odd
			}

			// Find the matching option (case-insensitive)
			int idx = findOption(key);
			if (idx < 0)
				continue;  // unknown option, skip

			wasSet[idx] = true;

			if (m_options[idx].type == BOOL_SWITCH)
			{
				values[idx] = "1";
			}
			else if (hasEquals)
			{
				values[idx] = val;
			}
			else if (m_options[idx].hasImplicit &&
			         (a + 1 >= args.size() || (!args[a+1].empty() && args[a+1][0] == '-')))
			{
				// Has implicit value and next arg is missing or is another option
				values[idx] = m_options[idx].implicitVal;
			}
			else if (a + 1 < args.size())
			{
				// Consume next argument as value
				values[idx] = args[++a];
			}
		}

		// Fire callbacks: first defaults (not explicitly set), then set values
		for (size_t i = 0; i < m_options.size(); ++i)
			fireCallback(m_options[i], values[i]);
	}

	/** Print formatted usage to stream. */
	void printUsage(std::ostream& os) const
	{
		if (!m_title.empty())
			os << m_title << ":\n";
		for (auto& o : m_options)
		{
			std::string names = "  --" + o.longName;
			if (o.shortName)
				names += std::string(", -") + o.shortName;
			os << names;
			if (names.size() < 28)
				os << std::string(28 - names.size(), ' ');
			else
				os << "\n" << std::string(28, ' ');
			os << o.description << "\n";
		}
	}

	bool empty() const { return m_options.empty(); }

private:
	enum Type { BOOL_SWITCH, STRING_VALUE, INT_VALUE, UINT_VALUE };

	struct Option
	{
		std::string longName;
		char shortName = 0;
		Type type = BOOL_SWITCH;
		std::string defaultVal;
		std::string implicitVal;
		bool hasImplicit = false;
		std::string description;
		std::function<void(bool)> boolCb;
		std::function<void(const std::string&)> stringCb;
		std::function<void(int)> intCb;
		std::function<void(unsigned int)> uintCb;
	};

	std::string m_title;
	std::vector<Option> m_options;

	static void parseName(const std::string& spec, std::string& longName, char& shortName)
	{
		auto comma = spec.find(',');
		if (comma != std::string::npos)
		{
			longName = spec.substr(0, comma);
			if (comma + 1 < spec.size())
				shortName = spec[comma + 1];
			else
				shortName = 0;
		}
		else
		{
			longName = spec;
			shortName = 0;
		}
	}

	static std::string toLower(const std::string& s)
	{
		std::string r = s;
		std::transform(r.begin(), r.end(), r.begin(),
		               [](unsigned char c){ return std::tolower(c); });
		return r;
	}

	int findOption(const std::string& key) const
	{
		std::string keyLower = toLower(key);
		// Try short name first (single char)
		if (keyLower.size() == 1)
		{
			char c = keyLower[0];
			for (size_t i = 0; i < m_options.size(); ++i)
				if (m_options[i].shortName && std::tolower(m_options[i].shortName) == c)
					return (int)i;
		}
		// Try long name (case insensitive)
		for (size_t i = 0; i < m_options.size(); ++i)
			if (toLower(m_options[i].longName) == keyLower)
				return (int)i;
		return -1;
	}

	static void fireCallback(const Option& o, const std::string& val)
	{
		switch (o.type)
		{
		case BOOL_SWITCH:
			if (o.boolCb) o.boolCb(val == "1");
			break;
		case STRING_VALUE:
			if (o.stringCb) o.stringCb(val);
			break;
		case INT_VALUE:
			if (o.intCb) o.intCb(std::stoi(val));
			break;
		case UINT_VALUE:
			if (o.uintCb) o.uintCb(static_cast<unsigned int>(std::stoul(val)));
			break;
		}
	}
};

#endif // OPTION_PARSER_H
