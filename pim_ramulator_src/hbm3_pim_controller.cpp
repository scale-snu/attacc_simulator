#include "dram_controller/controller.h"
#include "memory_system/memory_system.h"

namespace Ramulator {

class HBM3PIMController final : public IDRAMController, public Implementation {
  RAMULATOR_REGISTER_IMPLEMENTATION(IDRAMController, HBM3PIMController, "HBM3-PIM", "A HBM3-PIM controller.");
  private:
    std::deque<Request> pending;          // A queue for read requests that are about to finish (callback after RL)

    ReqBuffer m_active_buffer;            // Buffer for requests being served. This has the highest priority 
    ReqBuffer m_priority_buffer;          // Buffer for high-priority requests (e.g., maintenance like refresh).
    ReqBuffer m_read_buffer;              // Read request buffer
    ReqBuffer m_write_buffer;             // Write request buffer
    ReqBuffer m_pim_buffer;               // PIM request buffer (in-order, higher priority than read/write buffer, lower priority than priority buffer)

    int m_row_addr_idx = -1;

    float m_wr_low_watermark;
    float m_wr_high_watermark;
    bool  m_is_write_mode = false;

    std::vector<IControllerPlugin*> m_plugins;

    size_t s_num_row_hits = 0;
    size_t s_num_row_misses = 0;
    size_t s_num_row_conflicts = 0;

    bool is_pim = false;

  public:
    void init() override {
      m_wr_low_watermark =  param<float>("wr_low_watermark").desc("Threshold for switching back to read mode.").default_val(0.2f);
      m_wr_high_watermark = param<float>("wr_high_watermark").desc("Threshold for switching to write mode.").default_val(0.8f);

      m_scheduler = create_child_ifce<IScheduler>();
      m_refresh = create_child_ifce<IRefreshManager>();    

      if (m_config["plugins"]) {
        YAML::Node plugin_configs = m_config["plugins"];
        for (YAML::iterator it = plugin_configs.begin(); it != plugin_configs.end(); ++it) {
          m_plugins.push_back(create_child_ifce<IControllerPlugin>(*it));
        }
      }
    };

    void setup(IFrontEnd* frontend, IMemorySystem* memory_system) override {
      m_dram = memory_system->get_ifce<IDRAM>();
      m_row_addr_idx = m_dram->m_levels("row");
      m_priority_buffer.max_size = 512*3 + 32;
    };

    bool send(Request& req) override {
      req.final_command = m_dram->m_request_translations(req.type_id);

      // Forward existing write requests to incoming read requests
      if (req.type_id == Request::Type::Read) {
        auto compare_addr = [req](const Request& wreq) {
          return wreq.addr == req.addr;
        };
        if (std::find_if(m_write_buffer.begin(), m_write_buffer.end(), compare_addr) != m_write_buffer.end()) {
          // The request will depart at the next cycle
          req.depart = m_clk + 1;
          pending.push_back(req);
          return true;
        }
      }

      // Else, enqueue them to corresponding buffer based on request type id
      bool is_success = false;
      req.arrive = m_clk;
      switch (req.type_id) {
        case Request::Type::Read:           is_success = m_read_buffer.enqueue(req);  break;
        case Request::Type::Write:          is_success = m_write_buffer.enqueue(req); break;
        case Request::Type::PIM_MAC_AB:     is_success = m_pim_buffer.enqueue(req);   break;
        case Request::Type::PIM_MAC_SB:     is_success = m_pim_buffer.enqueue(req);   break;
        case Request::Type::PIM_MAC_PB:     is_success = m_pim_buffer.enqueue(req);   break;
        case Request::Type::PIM_WR_GB:      is_success = m_pim_buffer.enqueue(req);   break;
        case Request::Type::PIM_MV_SB:      is_success = m_pim_buffer.enqueue(req);   break;
        case Request::Type::PIM_MV_GB:      is_success = m_pim_buffer.enqueue(req);   break;
        case Request::Type::PIM_SFM:        is_success = m_pim_buffer.enqueue(req);   break;
        case Request::Type::PIM_SET_MODEL:  is_success = m_pim_buffer.enqueue(req);   break;
        case Request::Type::PIM_SET_HEAD:   is_success = m_pim_buffer.enqueue(req);   break;
        case Request::Type::PIM_BARRIER:    is_success = m_pim_buffer.enqueue(req);   break;
        default: throw std::runtime_error("Invalid request type!");
      }

      if (!is_success) {
        // We could not enqueue the request
        req.arrive = -1;
        return false;
      }

      return true;
    };

    bool priority_send(Request& req) override {
      req.final_command = m_dram->m_request_translations(req.type_id);

      bool is_success = false;
      is_success = m_priority_buffer.enqueue(req);
      return is_success;
    }

    void tick() override {
      m_clk++;

      // 1. Serve completed reads
      serve_completed_reads();

      m_refresh->tick();

      // 2. Try to find a request to serve.
      ReqBuffer::iterator req_it;
      ReqBuffer* buffer = nullptr;
      bool request_found = schedule_request(req_it, buffer);

      // 3. Update all plugins
      for (auto plugin : m_plugins) {
        plugin->update(request_found, req_it);
      }

      // 4. Finally, issue the commands to serve the request
      if (request_found) {
        // If we find a real request to serve
        m_dram->issue_command(req_it->command, req_it->addr_vec);

        // If we are issuing the last command, set depart clock cycle and move the request to the pending queue
        if (req_it->command == req_it->final_command) {
          if (req_it->type_id == Request::Type::Read) {
            req_it->depart = m_clk + m_dram->m_read_latency;
            pending.push_back(*req_it);
          } else if (req_it->type_id == Request::Type::Write) {
            // TODO: Add code to update statistics
          }
          buffer->remove(req_it);
        } else {
          if (!is_pim) {
            if (m_dram->m_command_meta(req_it->command).is_opening) {
              m_active_buffer.enqueue(*req_it);
              buffer->remove(req_it);
            }
          }
        }
      }

      // 5. If next command is ROW/COL command while issued command is COL/ROW command, issue next command concurrently
      ReqBuffer::iterator sec_req_it;
      ReqBuffer* sec_buffer = nullptr;
      bool sec_request_found = false;
      sec_request_found = schedule_sec_request(sec_req_it, buffer, req_it->command);
      for (auto plugin : m_plugins) {
        plugin->update(sec_request_found, sec_req_it);
      }
      if (sec_request_found) {
        m_dram->issue_command(sec_req_it->command, sec_req_it->addr_vec);

        if (sec_req_it->command == sec_req_it->final_command) {
          if (sec_req_it->type_id == Request::Type::Read) {
            sec_req_it->depart = m_clk + m_dram->m_read_latency;
            pending.push_back(*sec_req_it);
          } else if (sec_req_it->type_id == Request::Type::Write) {
          }
          buffer->remove(sec_req_it);
        } else {
          if (!is_pim) {
            if (m_dram->m_command_meta(sec_req_it->command).is_opening) {
              m_active_buffer.enqueue(*sec_req_it);
              buffer->remove(sec_req_it);
            }
          }
        }
      }
    };


    bool is_pending() override {
      bool is_pending = m_active_buffer.size() ||  m_priority_buffer.size() || m_read_buffer.size() || m_write_buffer.size() || m_pim_buffer.size() || pending.size();
      return is_pending;
    };

  private:
    /**
     * @brief    Helper function to serve the completed read requests
     * @details
     * This function is called at the beginning of the tick() function.
     * It checks the pending queue to see if the top request has received data from DRAM.
     * If so, it finishes this request by calling its callback and poping it from the pending queue.
     */
    void serve_completed_reads() {
      if (pending.size()) {
        // Check the first pending request
        auto& req = pending[0];
        if (req.depart <= m_clk) {
          // Request received data from dram
          if (req.depart - req.arrive > 1) {
            // Check if this requests accesses the DRAM or is being forwarded.
            // TODO add the stats back
          }

          if (req.callback) {
            // If the request comes from outside (e.g., processor), call its callback
            req.callback(req);
          }
          // Finally, remove this request from the pending queue
          pending.pop_front();
        }
      };
    };


    /**
     * @brief    Checks if we need to switch to write mode
     * 
     */
    void set_write_mode() {
      if (!m_is_write_mode) {
        if ((m_write_buffer.size() > m_wr_high_watermark * m_write_buffer.max_size) || m_read_buffer.size() == 0) {
          m_is_write_mode = true;
        }
      } else {
        if ((m_write_buffer.size() < m_wr_low_watermark * m_write_buffer.max_size) && m_read_buffer.size() != 0) {
          m_is_write_mode = false;
        }
      }
    };


    /**
     * @brief    Helper function to find a request to schedule from the buffers.
     * 
     */
    bool schedule_request(ReqBuffer::iterator& req_it, ReqBuffer*& req_buffer) {
      bool request_found = false;
      // 2.1    First, check the act buffer to serve requests that are already activating (avoid useless ACTs)
      if (req_it= m_scheduler->get_best_request(m_active_buffer); req_it != m_active_buffer.end()) {
        if (m_dram->check_ready(req_it->command, req_it->addr_vec)) {
          request_found = true;
          req_buffer = &m_active_buffer;
        }
      }

      // 2.2    If no requests can be scheduled from the act buffer, check the rest of the buffers
      if (!request_found) {
        // 2.2.1    We first check the priority buffer to prioritize e.g., maintenance requests
        if (m_priority_buffer.size() != 0) {
          req_buffer = &m_priority_buffer;
          req_it = m_priority_buffer.begin();
          req_it->command = m_dram->get_preq_command(req_it->final_command, req_it->addr_vec);
          
          request_found = m_dram->check_ready(req_it->command, req_it->addr_vec);
          if (!request_found & m_priority_buffer.size() != 0) {
            return false;
          }
        }

        is_pim = false;
        // 2.2.2    If no request to be scheduled in the priority buffer, check the pim buffer for PIM operations.
        if (!request_found) {
          auto& buffer = m_pim_buffer;
          if (req_it = m_scheduler->get_best_request(buffer); req_it != buffer.end()) {
            request_found = m_dram->check_ready(req_it->command, req_it->addr_vec);
            req_buffer = &buffer;
          }
          if (request_found) {
            is_pim = true;
          }
        }


        // 2.2.3    If no request to be scheduled in the priority buffer, check the read and write buffers.
        if (!request_found) {
          // Query the write policy to decide which buffer to serve
          set_write_mode();
          auto& buffer = m_is_write_mode ? m_write_buffer : m_read_buffer;
          if (req_it = m_scheduler->get_best_request(buffer); req_it != buffer.end()) {
            request_found = m_dram->check_ready(req_it->command, req_it->addr_vec);
            req_buffer = &buffer;
          }
        }
      }

      // 2.3 If we find a request to schedule, we need to check if it will close an opened row in the active buffer.
      if (request_found) {
        if (m_dram->m_command_meta(req_it->command).is_closing) {
          std::vector<Addr_t> rowgroup((req_it->addr_vec).begin(), (req_it->addr_vec).begin() + m_row_addr_idx);

          // Search the active buffer with the row address (inkl. banks, etc.)
          for (auto _it = m_active_buffer.begin(); _it != m_active_buffer.end(); _it++) {
            std::vector<Addr_t> _it_rowgroup(_it->addr_vec.begin(), _it->addr_vec.begin() + m_row_addr_idx);
            if (rowgroup == _it_rowgroup) {
              // Invalidate this scheduling outcome if we are to interrupt a request in the active buffer
              request_found = false;
            }
          }
        }
      }

      return request_found;
    }


    /**
     * @brief    Helper function to find a second request to schedule from the buffers.
     * 
     */
    bool schedule_sec_request(ReqBuffer::iterator& req_it, ReqBuffer*& req_buffer, int first_command) {
      bool request_found = false;
      // 5.1    First, check the act buffer to serve requests that are already activating (avoid useless ACTs)
      if (req_it= m_scheduler->get_best_request(m_active_buffer); req_it != m_active_buffer.end()) {
        if (m_dram->check_ready(req_it->command, req_it->addr_vec)) {
          if (compare_command_type(first_command, req_it->command)) {
            request_found = true;
            req_buffer = &m_active_buffer;
          }
        }
      }

      // 5.2    If no requests can be scheduled from the act buffer, check the rest of the buffers
      if (!request_found) {
        // 5.2.1    We first check the priority buffer to prioritize e.g., maintenance requests
        if (m_priority_buffer.size() != 0) {
          req_buffer = &m_priority_buffer;
          req_it = m_priority_buffer.begin();
          req_it->command = m_dram->get_preq_command(req_it->final_command, req_it->addr_vec);
          
          if (compare_command_type(first_command, req_it->command)) {
            request_found = m_dram->check_ready(req_it->command, req_it->addr_vec);
            if (!request_found & m_priority_buffer.size() != 0) {
              return false;
            }
          }
        }

        is_pim = false;
        // 2.2.2    If no request to be scheduled in the priority buffer, check the pim buffer for PIM operations.
        if (!request_found) {
          auto& buffer = m_pim_buffer;
          if (req_it = m_scheduler->get_best_request(buffer); req_it != buffer.end()) {
            if (compare_command_type(first_command, req_it->command)) {
              request_found = m_dram->check_ready(req_it->command, req_it->addr_vec);
              req_buffer = &buffer;
            }
          }
          if (request_found) {
            is_pim = true;
          }
        }


        // 5.2.2    If no request to be scheduled in the priority buffer, check the read and write buffers.
        if (!request_found) {
          // Query the write policy to decide which buffer to serve
          set_write_mode();
          auto& buffer = m_is_write_mode ? m_write_buffer : m_read_buffer;
          if (req_it = m_scheduler->get_best_request(buffer); req_it != buffer.end()) {
            if (compare_command_type(first_command, req_it->command)) {
              request_found = m_dram->check_ready(req_it->command, req_it->addr_vec);
              req_buffer = &buffer;
            }
          }
        }
      }

      // 5.3 If we find a request to schedule, we need to check if it will close an opened row in the active buffer.
      if (request_found) {
        if (m_dram->m_command_meta(req_it->command).is_closing) {
          std::vector<Addr_t> rowgroup((req_it->addr_vec).begin(), (req_it->addr_vec).begin() + m_row_addr_idx);

          // Search the active buffer with the row address (inkl. banks, etc.)
          for (auto _it = m_active_buffer.begin(); _it != m_active_buffer.end(); _it++) {
            std::vector<Addr_t> _it_rowgroup(_it->addr_vec.begin(), _it->addr_vec.begin() + m_row_addr_idx);
            if (rowgroup == _it_rowgroup) {
              // Invalidate this scheduling outcome if we are to interrupt a request in the active buffer
              request_found = false;
            }
          }
        }
      }

      return request_found;
    }

    // Row command: 0,  Col command: 1
    int get_command_type(int command) {
      if      (command == m_dram->m_commands("ACT"))   return 0; // Row command
      else if (command == m_dram->m_commands("PRE"))   return 0; // Row command
      else if (command == m_dram->m_commands("PREA"))  return 0; // Row command
      else if (command == m_dram->m_commands("RD"))    return 1; // Col command
      else if (command == m_dram->m_commands("WR"))    return 1; // Col command
      else if (command == m_dram->m_commands("REFab")) return 0; // Row command
      else if (command == m_dram->m_commands("REFsb")) return 0; // Row command
      else if (command == m_dram->m_commands("ACTAB")) return 0; // Row command
      else if (command == m_dram->m_commands("ACTSB")) return 0; // Row command
      else if (command == m_dram->m_commands("ACTPB")) return 0; // Row command
      else if (command == m_dram->m_commands("MACAB")) return 1; // Col command
      else if (command == m_dram->m_commands("MACSB")) return 1; // Col command
      else if (command == m_dram->m_commands("MACPB")) return 1; // Col command
      else if (command == m_dram->m_commands("WRGB"))  return 1; // Col command
      else if (command == m_dram->m_commands("MVSB"))  return 1; // Col command
      else if (command == m_dram->m_commands("MVGB"))  return 1; // Col command
      else if (command == m_dram->m_commands("SFM"))   return 1; // Col command
      else if (command == m_dram->m_commands("SETM"))  return 1; // Col command
      else if (command == m_dram->m_commands("SETH"))  return 1; // Col command
      else return -1;
    }

    bool compare_command_type(int first_command, int second_command) {
      int first_command_type = get_command_type(first_command);
      int second_command_type = get_command_type(second_command);
 
      if (first_command_type == -1 || second_command_type == -1) {
        return false;
      }

      if (first_command_type == second_command_type) {
        return false;
      }
      else {
        return true;
      }
    }
};
  
}   // namespace Ramulator
